from __future__ import annotations
import ast
import dataclasses
import torch
import torch.nn as nn
import z3
from z3 import Int, Solver, sat, unsat, And, Or, Not

_ALLOWED_CMP = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def _floordiv(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a // b
    return a / b


_ALLOWED_BINOP = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: _floordiv,
}


class UnsafeExpressionError(ValueError):
    pass


def _compile_expr(node: ast.AST, vars_: dict[str, z3.ArithRef]):
    if isinstance(node, ast.Expression):
        return _compile_expr(node.body, vars_)
    if isinstance(node, ast.BoolOp):
        values = [_compile_expr(v, vars_) for v in node.values]
        if isinstance(node.op, ast.And):
            return And(*values)
        if isinstance(node.op, ast.Or):
            return Or(*values)
        raise UnsafeExpressionError(f"Disallowed boolean operator: {node.op}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return Not(_compile_expr(node.operand, vars_))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_compile_expr(node.operand, vars_)
    if isinstance(node, ast.Compare):
        left = _compile_expr(node.left, vars_)
        result = None
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _compile_expr(comparator, vars_)
            if type(op) not in _ALLOWED_CMP:
                raise UnsafeExpressionError(f"Disallowed comparison operator: {op}")
            term = _ALLOWED_CMP[type(op)](left, right)
            result = term if result is None else And(result, term)
            left = right
        return result
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BINOP:
            raise UnsafeExpressionError(
                f"Disallowed arithmetic operator: {node.op}"
            )
        return _ALLOWED_BINOP[type(node.op)](
            _compile_expr(node.left, vars_), _compile_expr(node.right, vars_)
        )
    if isinstance(node, ast.Name):
        if node.id not in vars_:
            raise UnsafeExpressionError(f"Unknown variable: {node.id}")
        return vars_[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    raise UnsafeExpressionError(f"Disallowed AST node: {type(node).__name__}")


def safe_z3_expr(expr_str: str, vars_: dict[str, z3.ArithRef]):
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"Syntax error in '{expr_str}': {e}") from e
    return _compile_expr(tree, vars_)


@dataclasses.dataclass
class SymbolicSpec:
    var_names: list[str]
    constraints: list[str]
    description: str = ""


@dataclasses.dataclass
class Z3CheckResult:
    status: str
    message: str
    model_values: dict[str, int] | None = None
    unsat_core: list[str] | None = None


class Z3LogicalEngine:

    @staticmethod
    def solve_bounds_check(var_name: str, lower_bound: int, upper_bound: int) -> str:
        s = Solver()
        x = Int(var_name)
        s.add(x > lower_bound, x < upper_bound)
        if s.check() == sat:
            model = s.model()
            return f"SAT: example value {var_name} = {model[x]}"
        return "UNSAT: logical contradiction"

    @staticmethod
    def check_spec(spec: SymbolicSpec) -> Z3CheckResult:
        try:
            z3_vars = {name: Int(name) for name in spec.var_names}
            s = Solver()
            for i, c in enumerate(spec.constraints):
                expr = safe_z3_expr(c, z3_vars)
                s.assert_and_track(expr, f"c{i}")
            result = s.check()
            if result == sat:
                model = s.model()
                values = {
                    name: model[var].as_long()
                    for name, var in z3_vars.items()
                    if model[var] is not None
                }
                return Z3CheckResult(
                    status="sat",
                    message=f"SAT: found a model satisfying all constraints: {values}",
                    model_values=values,
                )
            elif result == unsat:
                core = [str(c) for c in s.unsat_core()]
                offending = [
                    spec.constraints[int(c[1:])]
                    for c in core
                    if c.startswith("c") and c[1:].isdigit()
                ]
                return Z3CheckResult(
                    status="unsat",
                    message=f"UNSAT: the following constraints are mutually contradictory and CANNOT be satisfied simultaneously: {offending}. Remove or relax one of them.",
                    unsat_core=offending,
                )
            else:
                return Z3CheckResult(
                    status="error", message=f"Z3 returned an unknown status: {result}"
                )
        except UnsafeExpressionError as e:
            return Z3CheckResult(
                status="error", message=f"Invalid constraint specification: {e}"
            )
        except Exception as e:
            return Z3CheckResult(status="error", message=f"Z3 error: {e}")


class SymbolicTrigger(nn.Module):

    def __init__(self, dim: int, threshold: float = 0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )
        self.threshold = threshold

    def forward(self, latent_concepts: torch.Tensor) -> torch.Tensor:
        pooled_logits = self.classifier(latent_concepts).squeeze(-1)
        return torch.sigmoid(pooled_logits)

    def should_call_z3(self, latent_concepts: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = self.forward(latent_concepts)
        return probs > self.threshold
