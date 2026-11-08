import ast
from typing import cast


class NKITransformer(ast.NodeTransformer):
    """
    AST transformer for NKI simulation.

    Transformations:
    - `nl.<func>(x[...], ...)` -> `nl.<func>(x, slice_obj, ...)`
       where <func> is one of {load, store, load_transpose2d}, wherever the call
       appears (e.g. nested in another call, as in `nl.store(o[i], nl.load(x[i]))`)
    - `nisa.dma_copy(dst=o[...], src=x[...])` and `nisa.dma_transpose(x[...])` -> keys passed
       as `dst_keys` / `src_keys`
    - strips out kernel decorators
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.decorator_list = []
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = cast(ast.Call, self.generic_visit(node))
        new_call = self._rewrite_nl_slice_call(node)
        if new_call is None:
            new_call = self._rewrite_dma_copy_call(node)
        return node if new_call is None else new_call

    def _rewrite_dma_copy_call(self, call_node: ast.Call) -> ast.Call | None:
        """`nisa.dma_copy(dst=o[...], src=x[...])` -> `nisa.dma_copy(dst=o, dst_keys=..., src=x, src_keys=...)`"""
        if not (
            isinstance(call_node.func, ast.Attribute)
            and isinstance(call_node.func.value, ast.Name)
            and call_node.func.value.id == "nisa"
            and call_node.func.attr in ("dma_copy", "dma_transpose")
        ):
            return None
        args = list(call_node.args)
        keywords = []
        changed = False
        if (
            call_node.func.attr == "dma_transpose"
            and args
            and isinstance(args[0], ast.Subscript)
        ):
            # `nisa.dma_transpose(x[...])` -> `nisa.dma_transpose(x, src_keys=...)`
            keywords.append(
                ast.keyword(
                    arg="src_keys",
                    value=cast(ast.expr, self._create_slice_value_node(args[0].slice)),
                )
            )
            args[0] = args[0].value
            changed = True
        for kw in call_node.keywords:
            if kw.arg in ("dst", "src") and isinstance(kw.value, ast.Subscript):
                keywords.append(ast.keyword(arg=kw.arg, value=kw.value.value))
                keywords.append(
                    ast.keyword(
                        arg=f"{kw.arg}_keys",
                        value=cast(
                            ast.expr, self._create_slice_value_node(kw.value.slice)
                        ),
                    )
                )
                changed = True
            else:
                keywords.append(kw)
        if not changed:
            return None
        return ast.Call(func=call_node.func, args=args, keywords=keywords)

    def _rewrite_nl_slice_call(self, call_node: ast.Call) -> ast.Call | None:
        if not (
            isinstance(call_node.func, ast.Attribute)
            and isinstance(call_node.func.value, ast.Name)
            and call_node.func.value.id == "nl"
            and call_node.func.attr in ("store", "load", "load_transpose2d")
        ):
            return None

        if not call_node.args or not isinstance(call_node.args[0], ast.Subscript):
            return None

        slice_node = call_node.args[0]
        return ast.Call(
            func=call_node.func,
            args=[
                slice_node.value,
                cast(ast.expr, self._create_slice_value_node(slice_node.slice)),
                *call_node.args[1:],
            ],
            keywords=call_node.keywords,
        )

    def _create_slice_value_node(self, node: ast.AST) -> ast.AST:
        def process_key(key: ast.AST):
            if isinstance(key, ast.Slice):
                return ast.Call(
                    func=ast.Name(id="slice", ctx=ast.Load()),
                    args=[
                        key.lower or ast.Constant(value=None),
                        key.upper or ast.Constant(value=None),
                        key.step or ast.Constant(value=None),
                    ],
                    keywords=[],
                )
            return key

        if isinstance(node, ast.Tuple):
            return ast.Tuple(elts=[process_key(e) for e in node.elts], ctx=ast.Load())
        return process_key(node)


def transform_code(source_code: str) -> str:
    tree = ast.parse(source_code)
    transformer = NKITransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
