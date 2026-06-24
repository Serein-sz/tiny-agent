use std::io::{Write, stdout};

use async_openai::{
    Client,
    types::chat::{
        ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
        ChatCompletionRequestUserMessage, ChatCompletionTool, CreateChatCompletionRequestArgs,
        FinishReason, FunctionObjectArgs,
    },
};
use futures::StreamExt;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::info;

use crate::constant::MODEL;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WritePlanOptions {
    path: String,
    tasks: Vec<Task>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Task {
    index: usize,
    content: String,
    difficulty: String,
}

pub async fn completion_stream(
    mut messages: Vec<ChatCompletionRequestMessage>,
) -> anyhow::Result<()> {
    for i in 0..5 {
        info!("loop: {}", i + 1);
        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(2048u32)
            .model(MODEL)
            .tools(ChatCompletionTool {
                function: FunctionObjectArgs::default()
                    .name("write_todo")
                    .description("write task todo")
                    .parameters(serde_json::to_value(schema_for!(WritePlanOptions))?)
                    .strict(true)
                    .build()?,
            })
            .messages(messages.clone())
            .build()?;
        let client = Client::new();
        let mut stream = client.chat().create_stream(request).await?;
        let mut tool_calls = vec![];
        let mut execution_handles = vec![];
    while let Some(response) = stream.next().await {
            let chunk = response?;
            for choice in chunk.choices {
                if let Some(content) = choice.delta.content {
                    let mut stdout_lock = stdout().lock();
                    write!(stdout_lock, "{}", content)?;
                    stdout_lock.flush()?;
                }
                if let Some(tool_call_chunks) = choice.delta.tool_calls {
                    for chunk in tool_call_chunks.iter() {
                        let index = chunk.index as usize;
                        if tool_calls.len() <= index {
                            tool_calls.push(ChatCompletionMessageToolCall {
                                id: Default::default(),
                                function: Default::default(),
                            });
                        }
                        let tool_call = &mut tool_calls[index];
                        if let Some(id) = &chunk.id {
                            tool_call.id = id.clone();
                        }
                        if let Some(function_call_stream) = &chunk.function {
                            if let Some(name) = &function_call_stream.name {
                                tool_call.function.name = name.clone();
                            }
                            if let Some(arguments) = &function_call_stream.arguments {
                                tool_call.function.arguments.push_str(arguments);
                            }
                        }
                    }
                }
                if matches!(choice.finish_reason, Some(FinishReason::ToolCalls)) {
                    for tool_call in tool_calls.iter() {
                        let id = tool_call.id.clone();
                        let name = tool_call.function.name.clone();
                        let args = tool_call.function.arguments.clone();
                        let handle =
                            tokio::spawn(async move { (id, call_function(&name, &args).await) });
                        execution_handles.push(handle);
                    }
                }
            }
        }

        if execution_handles.is_empty() {
            break;
        }
        let mut tool_responses = Vec::new();
        for handle in execution_handles {
            let (tool_call_id, response) = handle.await?;
            tool_responses.push((tool_call_id, response));
        }
        // Build the follow-up request using ergonomic From traits
        // Add assistant message with tool calls
        let assistant_tool_calls: Vec<ChatCompletionMessageToolCalls> = tool_calls
            .iter()
            .map(|tc| tc.clone().into()) // From<ChatCompletionMessageToolCall>
            .collect();
        messages.push(
            ChatCompletionRequestAssistantMessage {
                content: None,
                tool_calls: Some(assistant_tool_calls),
                ..Default::default()
            }
            .into(),
        );
        // Add tool response messages
        for (tool_call_id, response) in tool_responses {
            messages.push(
                ChatCompletionRequestToolMessage {
                    content: response?.to_string().into(),
                    tool_call_id,
                }
                .into(),
            );
        }
    }
    Ok(())
}

async fn call_function(name: &str, args: &str) -> anyhow::Result<serde_json::Value> {
    match name {
        "write_todo" => write_todo(serde_json::from_str::<WritePlanOptions>(args)?).await,
        _ => Ok(json!({
            "success": false,
            "error": format!("Unknown function: {}", name)
        })),
    }
}

async fn write_todo(options: WritePlanOptions) -> anyhow::Result<serde_json::Value> {
    // 这里先打印看看
    info!("write_todo options: {:#?}", options);

    Ok(json!({
        "success": true,
        "path": options.path,
        "task_count": options.tasks.len()
    }))
}
pub async fn completion() -> anyhow::Result<()> {
    info!("completion enter");
    completion_stream(vec![
        ChatCompletionRequestSystemMessage::from(
            "你是一个高级Agent开发工程师. Guide the user through the solution step by step.",
        )
        .into(),
        ChatCompletionRequestUserMessage::from("我想构建一个coding agent 请给出简略计划").into(),
    ])
    .await?;

    Ok(())
}
