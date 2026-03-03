use runtime::LunaRuntime;

fn main() {
    println!("🌙 Luna - AI Code Assistant");
    println!("Type 'exit' to quit\n");

    let _runtime = LunaRuntime::new();

    loop {
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        match input {
            "exit" | "quit" => {
                println!("Goodbye!");
                break;
            }
            "" => continue,
            _ => {
                println!("Received: {}", input);
                println!("(Runtime not yet implemented)\n");
            }
        }
    }
}
