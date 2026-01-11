from mcp.server import FastMCP

app = FastMCP("hybridRag")

@app.tool('add')
def add_tool(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

if __name__ == '__main__':
    app.run('stdio')
