use anyhow::Result;
use intelligence::TreeSitterFile;

fn main() -> Result<()> {
    println!("Luna Intelligence Demo\n");

    // 1. 准备一段测试代码 (Rust)
    let code = r#"
    fn add(a: i32, b: i32) -> i32 {
        return a + b;
    }

    fn main() {
        let result = add(1, 2);
        println!("Result: {}", result);
    }
    "#;

    println!(" Analyzing Source Code:\n---\n{}\n---", code);

    // 2. 使用 Intelligence 模块进行解析
    // "Rust" 是语言 ID，对应 xc-intelligence 内部的注册
    let ts_file = TreeSitterFile::try_build(code.as_bytes(), "Rust")
        .map_err(|e| anyhow::anyhow!("Failed to parse: {:?}", e))?;

    // 3. 获取 Scope Graph (核心能力：理解作用域、定义和引用)
    let scope_graph = ts_file
        .scope_graph()
        .map_err(|e| anyhow::anyhow!("Failed to build scope graph: {:?}", e))?;

    // 4. 打印所有识别到的符号 (定义)
    println!("\n Detected Symbols (Definitions):");
    let symbols = scope_graph.symbols();

    if symbols.is_empty() {
        println!("   (No symbols found - check query files)");
    } else {
        for symbol in symbols {
            // format: [Line:Column] Kind - Name? (Name extraction might need source slicing)
            let name_range = symbol.range;
            let name = &code[name_range.start.byte..name_range.end.byte];
            println!(
                "    line {}:{} \t[{}] \t{}",
                name_range.start.line + 1,
                name_range.start.column + 1,
                symbol.kind,
                name
            );
        }
    }

    println!("\n Demo finished successfully.");
    Ok(())
}
