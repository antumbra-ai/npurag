use std::net::TcpListener;

/// Accept connections and hand each one to the worker pool.
fn main() -> std::io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:7878")?;
    for stream in listener.incoming() {
        let stream = stream?;
        handle(stream);
    }
    Ok(())
}

fn handle(_stream: std::net::TcpStream) {
    // The worker pool arrives later.
}
