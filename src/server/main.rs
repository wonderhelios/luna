mod handlers;
mod rpc;
mod server;
mod session;
mod util;
mod virtual_tools;

fn main() -> anyhow::Result<()> {
    server::run()
}
