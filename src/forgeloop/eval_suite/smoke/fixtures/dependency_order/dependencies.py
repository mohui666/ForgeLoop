def dependency_order(graph: dict[str, list[str]]) -> list[str]:
    result = []
    visited = set()

    def visit(node):
        if node in visited:
            return
        visited.add(node)
        result.append(node)
        for dependency in graph.get(node, []):
            visit(dependency)

    for node in graph:
        visit(node)
    return result
