use dotenv::dotenv;

use tiny_agent::trace::init_tracing;
use tiny_agent::core::chat;




#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    init_tracing();
    
    chat::completion().await?;

    Ok(())
}
