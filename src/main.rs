use tracing::info;

use async_openai::{
    Client,
    types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage, CreateChatCompletionRequestArgs, ResponseFormat
    },
};
use dotenv::dotenv;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use tiny_agent::constant::MODEL;
use tiny_agent::trace::init_tracing;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub index: String,
    pub explanation: String,
    pub use_time: String
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ActionPlan {
    pub goal: String,
    pub steps: Vec<Step>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StructuredOutput {
    /// A description of what the response format is for, used by the model to determine how to respond in the format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The name of the response format. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    pub name: String,
    /// The schema for the response format, described as a JSON Schema object.
    /// Learn how to build JSON schemas [here](https://json-schema.org/).
    pub schema: serde_json::Value,
    /// Whether to enable strict schema adherence when generating the output.
    /// If set to true, the model will always follow the exact schema defined
    /// in the `schema` field. Only a subset of JSON Schema is supported when
    /// `strict` is `true`. To learn more, read the [Structured Outputs
    /// guide](https://platform.openai.com/docs/guides/structured-outputs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

pub async fn structured_output<T: serde::Serialize + DeserializeOwned + JsonSchema>(
    mut messages: Vec<ChatCompletionRequestMessage>,
) -> anyhow::Result<Option<T>> {
    let schema = schema_for!(T);
    let schema_value = serde_json::to_value(&schema)?;

    messages.insert(
        0,
        ChatCompletionRequestSystemMessage::from(serde_json::to_string(
            &StructuredOutput {
                name: "math_reasoning".into(),
                description: None,
                schema: schema_value,
                strict: Some(true),
            }
        ).unwrap().as_str())
        .into(),
    );

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(2048u32)
        .model(MODEL)
        .messages(messages)
        .response_format(ResponseFormat::JsonObject)
        .build()?;

    let client = Client::new();
    let response = client.chat().create(request).await?;
    // info!("response: {:#?}", response);
    for choice in response.choices {
        if let Some(content) = choice.message.content {
            return Ok(Some(serde_json::from_str::<T>(&content)?));
        }
    }

    Ok(None)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    init_tracing();

    if let Some(response) = structured_output::<ActionPlan>(vec![
        ChatCompletionRequestSystemMessage::from(
            "你是一个高级Agent开发工程师. Guide the user through the solution step by step.",
        )
        .into(),
        ChatCompletionRequestUserMessage::from("我想构建一个coding agent 请给出简略计划").into(),
    ])
    .await?
    {
        info!("{:#?}", &response);
    }

    Ok(())
}
