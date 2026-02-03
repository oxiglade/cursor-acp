use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    cursor_acp::run_main().await?;
    Ok(())
}
