# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Class for evaluating programs proposed by the Sampler."""
from __future__ import annotations

from abc import abstractmethod, ABC
import ast
import time
from collections.abc import Sequence
import copy
from typing import Any, Type
import profile
import multiprocessing
import os
import sys
import traceback
import re

from llmsr import code_manipulation
from llmsr import buffer
from llmsr import evaluator_accelerate

# Import AtomSymbolicAnalyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from atom_integration import AtomSymbolicAnalyzer
except ImportError:
    # Create a fallback if the module doesn't exist yet
    AtomSymbolicAnalyzer = None


class _FunctionLineVisitor(ast.NodeVisitor):
    """ Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None: 
        """ Collect the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """ Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None 
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    """ Extract the body of the generated function, trimming anything after it.
    Please note that the indentation is REQUIRED !!!
    """
    if not generated_code:
        return ''

    code = f'def fake_function_header():\n{generated_code}'

    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        
        except SyntaxError as e:
            if e.lineno is None: # Nothing could be saved when syntaxError
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

    if not code:
        return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
        generated_code: str,
        version_generated: int | None,
        template: code_manipulation.Program,
        function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """ 
    Return the compiled generated function and the full runnable program.
    This function removes the content after the generated function body.
    """
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    
    return evolved_function, str(program)


class Sandbox(ABC):
    """ Sandbox for executing generated code. """

    @abstractmethod
    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,  
            test_input: str, 
            timeout_seconds: int,
            **kwargs
    ) -> tuple[Any, bool]:
        """ Return `function_to_run(test_input)` and whether execution succeeded. """
        raise NotImplementedError(
            'Must provide a sandbox for executing untrusted code.')


class LocalSandbox(Sandbox):
    """
    Secure environment for executing and evaluating LLM generated programs.
    Prevents harmful operations, limits resource usage, and enforces timeouts.
    Returns a 'score' for the executed program.
    """

    def __init__(self, verbose=False, numba_accelerate=False):
        """
        Initialize Sandbox.
        
        Args:
        verbose (bool): Enable detailed output.
        numba_accelerate (bool): Use Numba for acceleration of evaluation (limited compatibility). 
        """
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate


    def run(self, program: str, function_to_run: str, function_to_evolve: str, 
        inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[Any, bool]:
        """
        Execute the given program sample and return its score and success status.
        
        Note: This sandbox is specific to the equation program skeleton discovery problem.
        """
        print("\n----- SANDBOX: Starting Program Execution -----")
        print(f"Function to run: {function_to_run}")
        print(f"Function to evolve: {function_to_evolve}")
        print(f"Timeout: {timeout_seconds} seconds")

        dataset = inputs[test_input]
        result_queue = multiprocessing.Queue()
        
        print("\nLaunching program in separate process...")
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
        )
        process.start()
        process.join(timeout=timeout_seconds)

        # if the process is not finished in time, terminate
        if process.is_alive():
            print("\nProgram Execution TIMEOUT - Terminating process")
            process.terminate()
            process.join()
            results = None, False
        else:
            print("\nProgram completed within timeout")
            results = self._get_results(result_queue)
            print(f"Execution results: {results}")
        
        if self._verbose:
            self._print_evaluation_details(program, results, **kwargs)

        print("----- SANDBOX: Program Execution Complete -----\n")
        return results


    def _get_results(self, queue):
        for _ in range(5):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(0.1)
        return None, False


    def _print_evaluation_details(self, program, results, **kwargs):
        print('================= Evaluated Program =================')
        function = code_manipulation.text_to_program(program).get_function(kwargs.get('func_to_evolve', 'equation'))
        print(f'{str(function).strip()}\n-----------------------------------------------------')
        print(f'Score: {results}\n=====================================================\n\n')



    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, 
                                  dataset, numba_accelerate, result_queue):
        try:
            print("\n----- Compiling and Running Function -----")
            # optimize the code (decorate function_to_run with @numba.jit())
            if numba_accelerate:
                print("Applying Numba optimization...")
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            
            print("Executing program...")
            # execute the program, map func/var/class to global namespace
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            results = function_to_run(dataset)
            
            if not isinstance(results, (int, float)):
                print("Error: Results not numeric")
                result_queue.put((None, False))
                return
            print(f"Execution successful - Result: {results}")
            result_queue.put((results, True))
            
        # if raise any exception, execution is failed
        except Exception as e:
            print(f"Execution Error: {e}")
            result_queue.put((None, False))



def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """ Return whether the generated function is calling an earlier version. """
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False



class Evaluator:
    """ Class that analyses functions generated by LLMs. """

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            template: code_manipulation.Program,
            function_to_evolve: str,
            function_to_run: str,
            inputs: Sequence[Any],
            timeout_seconds: int,
            sandbox_class: Type[Sandbox] = Sandbox,
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()
        
        # Initialize the Atom analyzer for functions with good scores
        self._atom_analyzer = None
        self._use_atom_analysis = os.environ.get('USE_ATOM_ANALYSIS', 'False').lower() == 'true'
        self._atom_analysis_threshold = float(os.environ.get('ATOM_ANALYSIS_THRESHOLD', '-0.001'))
        
        # Configuration for term contribution analysis
        self._use_term_analysis = os.environ.get('USE_TERM_ANALYSIS', 'True').lower() == 'true'
        
        if AtomSymbolicAnalyzer is not None:
            try:
                self._atom_analyzer = AtomSymbolicAnalyzer()
                print("Atom analyzer initialized but disabled by default")
                print("Set USE_ATOM_ANALYSIS=true to enable Atom analysis")
            except Exception as e:
                print(f"Failed to initialize Atom analyzer: {e}")
                self._use_atom_analysis = False
        
        # Print configuration status
        print(f"Term contribution analysis: {'ENABLED' if self._use_term_analysis else 'DISABLED'}")
        print("Set USE_TERM_ANALYSIS=false to disable term contribution analysis")

    def _get_op_symbol(self, op):
        """Get string representation of AST operator."""
        if isinstance(op, ast.Add):
            return '+'
        elif isinstance(op, ast.Sub):
            return '-'
        elif isinstance(op, ast.Mult):
            return '*'
        elif isinstance(op, ast.Div):
            return '/'
        elif isinstance(op, ast.Pow):
            return '**'
        return str(op)

    def _find_dependencies(self, expr, assignment_map):
        """Find variables that an expression depends on."""
        deps = set()
        try:
            expr_ast = ast.parse(expr, mode='eval').body
            for node in ast.walk(expr_ast):
                if isinstance(node, ast.Name) and node.id in assignment_map:
                    deps.add(node.id)
        except:
            pass
        return deps

    def _analyze_assignments(self, function_body):
        """Build dependency graph of variable assignments."""
        assignment_map = {}
        for line in function_body.splitlines():
            assign_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)', line.strip())
            if assign_match:
                var, expr = assign_match.groups()
                assignment_map[var] = {
                    'expr': expr,
                    'depends_on': self._find_dependencies(expr, assignment_map)
                }
        return assignment_map

    def _extract_terms(self, node, assignment_map=None):
        """Enhanced term extraction with support for function calls and complex expressions."""
        terms = []
        
        try:
            if isinstance(node, ast.BinOp):
                # Handle binary operations
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    terms.extend(self._extract_terms(node.left, assignment_map))
                    terms.extend(self._extract_terms(node.right, assignment_map))
                else:
                    # Keep multiplication, division, and power as single terms
                    terms.append(node)
            elif isinstance(node, ast.Call):
                # Handle function calls like np.sin, np.abs
                if isinstance(node.func, ast.Attribute):
                    # Handle numpy functions
                    terms.append(node)
                else:
                    terms.append(node)
            elif isinstance(node, ast.UnaryOp):
                # Handle negative terms
                terms.extend(self._extract_terms(node.operand, assignment_map))
            elif isinstance(node, ast.Name) and assignment_map and node.id in assignment_map:
                # Expand variables using assignment map
                try:
                    expr = assignment_map[node.id]['expr']
                    expr_ast = ast.parse(expr, mode='eval').body
                    terms.extend(self._extract_terms(expr_ast, assignment_map))
                except Exception as e:
                    print(f"[DEBUG] Error expanding variable {node.id}: {e}")
                    terms.append(node)
            else:
                terms.append(node)
        except Exception as e:
            print(f"[DEBUG] Error extracting terms from node {type(node).__name__}: {e}")
            # Add the node as-is rather than crashing
            terms.append(node)
        
        return terms

    def _ast_to_str(self, node, assignment_map=None):
        """Enhanced AST to string conversion with proper operator precedence."""
        try:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                # Handle numpy function calls
                func_name = f"{node.func.value.id}.{node.func.attr}"
                args = [self._ast_to_str(arg, assignment_map) for arg in node.args]
                return f"{func_name}({', '.join(args)})"
            elif isinstance(node, ast.BinOp):
                # Handle binary operations with proper precedence
                left = self._ast_to_str(node.left, assignment_map)
                right = self._ast_to_str(node.right, assignment_map)
                op = self._get_op_symbol(node.op)
                
                # Add parentheses based on operator precedence
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    return f"{left} {op} {right}"
                else:
                    left = f"({left})" if isinstance(node.left, ast.BinOp) else left
                    right = f"({right})" if isinstance(node.right, ast.BinOp) else right
                    return f"{left}{op}{right}"
            elif isinstance(node, ast.UnaryOp):
                # Handle unary operations (like negation)
                if isinstance(node.op, ast.USub):
                    return f"-{self._ast_to_str(node.operand, assignment_map)}"
                return self._ast_to_str(node.operand, assignment_map)
            elif isinstance(node, ast.Name) and assignment_map and node.id in assignment_map:
                # Optionally expand variables
                try:
                    expr = assignment_map[node.id]['expr']
                    return f"({expr})"  # Wrap in parentheses to maintain precedence
                except Exception as e:
                    print(f"[DEBUG] Error expanding variable {node.id} in _ast_to_str: {e}")
                    return node.id
            elif isinstance(node, ast.Constant):
                return str(node.value)
            elif isinstance(node, ast.Name):
                return node.id
            else:
                # Fallback for other node types
                try:
                    return ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                except Exception as e:
                    print(f"[DEBUG] Error in ast.unparse for node {type(node).__name__}: {e}")
                    return str(node)
        except Exception as e:
            print(f"[DEBUG] Error in _ast_to_str for node {type(node).__name__}: {e}")
            # Return a safe fallback rather than crashing
            return f"<error_node_{type(node).__name__}>"

    def _extract_problem_description(self) -> str:
        """Extract the problem description from the template."""
        try:
            # Extract description from docstring at the beginning of the template
            program_str = str(self._template)
            if '"""' in program_str:
                docstring_start = program_str.find('"""') + 3
                docstring_end = program_str.find('"""', docstring_start)
                if docstring_end > docstring_start:
                    return program_str[docstring_start:docstring_end].strip()
            
            # Alternative: find problem description in function docstring
            func = self._template.get_function(self._function_to_evolve)
            if func.docstring:
                return func.docstring.strip()
                
            return "No description available"
        except Exception:
            return "Failed to extract problem description"

    def _analyze_term_combinations(self, terms, function_body, return_expr, assignment_map, new_function, avg_score):
        """Analyze contributions of both individual terms and term pairs."""
        term_contributions = []
        pair_contributions = []
        
        try:
            # First analyze individual terms
            for i, term in enumerate(terms):
                try:
                    # Build new expression without this term
                    new_terms = [t for j, t in enumerate(terms) if j != i]
                    if not new_terms:
                        continue
                    
                    # Reconstruct expression with proper operator handling
                    new_expr = self._ast_to_str(new_terms[0], assignment_map)
                    for t in new_terms[1:]:
                        new_expr = f"{new_expr} + {self._ast_to_str(t, assignment_map)}"
                    
                    # Get contribution score
                    contribution = self._evaluate_modified_function(
                        new_expr, function_body, return_expr, assignment_map, new_function, avg_score
                    )
                    
                    term_str = self._ast_to_str(term, assignment_map)
                    term_contributions.append((term_str, contribution))
                except Exception as e:
                    print(f"[DEBUG] Error analyzing individual term {i}: {e}")
                    continue
            
            # Then analyze term pairs (limit to prevent excessive computation)
            max_pairs = min(20, len(terms) * (len(terms) - 1) // 2)  # Limit to 20 pairs max
            pair_count = 0
            
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    if pair_count >= max_pairs:
                        break
                        
                    try:
                        # Build new expression without this pair of terms
                        new_terms = [t for k, t in enumerate(terms) if k != i and k != j]
                        if not new_terms:
                            continue
                        
                        # Reconstruct expression without the pair
                        new_expr = self._ast_to_str(new_terms[0], assignment_map)
                        for t in new_terms[1:]:
                            new_expr = f"{new_expr} + {self._ast_to_str(t, assignment_map)}"
                        
                        # Get contribution score for the pair
                        contribution = self._evaluate_modified_function(
                            new_expr, function_body, return_expr, assignment_map, new_function, avg_score
                        )
                        
                        term1_str = self._ast_to_str(terms[i], assignment_map)
                        term2_str = self._ast_to_str(terms[j], assignment_map)
                        pair_contributions.append(((term1_str, term2_str), contribution))
                        pair_count += 1
                        
                    except Exception as e:
                        print(f"[DEBUG] Error analyzing term pair ({i}, {j}): {e}")
                        continue
                        
                if pair_count >= max_pairs:
                    break
                    
        except Exception as e:
            print(f"[DEBUG] Error in _analyze_term_combinations: {e}")
            # Return empty lists on error rather than crashing
        
        return term_contributions, pair_contributions

    def _evaluate_modified_function(self, new_expr, function_body, return_expr, assignment_map, new_function, avg_score):
        """Evaluate a modified version of the function with some terms removed."""
        try:
            # Replace in function body
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', return_expr) and return_expr in assignment_map:
                new_body = re.sub(rf"{return_expr}\s*=\s*.+", f"{return_expr} = {new_expr}", function_body)
            else:
                new_body = re.sub(r"return .+", f"return {new_expr}", function_body)
            
            # Validate the modified function body for basic syntax
            try:
                # Try to parse the modified function to check for syntax errors
                test_code = f"def test_func():\n{new_body}"
                ast.parse(test_code)
            except SyntaxError as e:
                print(f"[DEBUG] Modified function has syntax error: {e}")
                return avg_score  # Return full contribution if syntax is invalid
            
            # Create and evaluate modified function
            test_func = copy.deepcopy(new_function)
            test_func.body = new_body
            test_program = copy.deepcopy(self._template)
            test_program.get_function(self._function_to_evolve).body = new_body
            test_program_str = str(test_program)
            
            # Evaluate modified function
            test_scores = {}
            for current_input in self._inputs:
                try:
                    test_output, runs_ok = self._sandbox.run(
                        test_program_str, self._function_to_run, self._function_to_evolve, 
                        self._inputs, current_input, self._timeout_seconds
                    )
                    if runs_ok and not _calls_ancestor(test_program_str, self._function_to_evolve) and test_output is not None:
                        if not isinstance(test_output, (int, float)):
                            continue
                        test_scores[current_input] = test_output
                except Exception as e:
                    print(f"[DEBUG] Error evaluating modified function for input {current_input}: {e}")
                    continue
            
            # Calculate contribution
            if test_scores:
                test_avg = sum(test_scores.values()) / len(test_scores)
                return avg_score - test_avg
            else:
                return avg_score  # If removal breaks function, assume full contribution
                
        except Exception as e:
            print(f"[DEBUG] Error in _evaluate_modified_function: {e}")
            return avg_score  # Return full contribution on any error

    def analyse(
            self,
            sample: str,
            island_id: int | None,
            version_generated: int | None,
            **kwargs 
    ) -> None:
        """ Compile the hypothesis sample into a program and executes it on test inputs. """
        try:
            print("\n========== EVALUATOR: Starting Analysis ==========")
            print(f"Island ID: {island_id}")
            print(f"Version Generated: {version_generated}")
            print(f"Iteration: {kwargs.get('global_sample_nums', 'Unknown')}")

            new_function, program = _sample_to_program(
                sample, version_generated, self._template, self._function_to_evolve)
            print("\n----- Generated Program -----")
            print(str(new_function))
            print("----------------------------")

            scores_per_test = {}
            time_reset = time.time()
            
            print("\n----- Starting Test Execution -----")
            for current_input in self._inputs:
                print(f"\nTesting input: {current_input}")
                test_output, runs_ok = self._sandbox.run(
                    program, self._function_to_run, self._function_to_evolve, self._inputs, current_input,
                    self._timeout_seconds
                )
                print(f"Test result - Success: {runs_ok}, Output: {test_output}")

                if runs_ok and not _calls_ancestor(program, self._function_to_evolve) and test_output is not None:
                    if not isinstance(test_output, (int, float)):
                        print(f'Error: test_output is {test_output}')
                        raise ValueError('@function.run did not return an int/float score.')
                    scores_per_test[current_input] = test_output
                    print(f"Score recorded: {test_output}")

            evaluate_time = time.time() - time_reset
            print(f"\nEvaluation time: {evaluate_time:.2f} seconds")
            print(f"Final scores: {scores_per_test}")

            if scores_per_test:
                # breakpoint()
                avg_score = sum(scores_per_test.values()) / len(scores_per_test)
                print(f"\n===== ITERATION {kwargs.get('global_sample_nums', 'Unknown')} - ISLAND {island_id} =====")
                print(f"Function successfully evaluated with average score: {avg_score}")

                # --- Term Contribution Analysis and Annotation ---
                if self._use_term_analysis:
                    try:
                        print("[Term Contribution] Starting analysis...")
                        from copy import deepcopy
                        function_body = new_function.body
                        
                        # Basic validation of function body
                        if not function_body or not function_body.strip():
                            print("[Term Contribution] Function body is empty, skipping analysis")
                            return
                        
                        # Build comprehensive assignment map with dependencies
                        assignment_map = self._analyze_assignments(function_body)
                        print("[DEBUG] Variable assignments and dependencies:", assignment_map)
                        
                        # Find the return statement
                        return_match = re.search(r"return (.+)", function_body)
                        if not return_match:
                            print("[Term Contribution] No return statement found, skipping analysis")
                            return
                            
                        return_expr = return_match.group(1).strip()
                        
                        # Handle return variable expansion
                        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', return_expr) and return_expr in assignment_map:
                            print(f"[DEBUG] Found assignment for return variable '{return_expr}': {assignment_map[return_expr]['expr']}")
                            expr_to_analyze = assignment_map[return_expr]['expr']
                        else:
                            expr_to_analyze = return_expr
                        
                        print("[DEBUG] Expression to analyze:", expr_to_analyze)
                        
                        # Parse and extract terms with enhanced AST handling
                        try:
                            # Validate expression syntax before parsing
                            if not expr_to_analyze or not expr_to_analyze.strip():
                                print("[Term Contribution] Expression to analyze is empty, skipping")
                                return
                                
                            expr_ast = ast.parse(expr_to_analyze, mode='eval').body
                            terms = self._extract_terms(expr_ast, assignment_map)
                            
                            if not terms:
                                print("[Term Contribution] No terms extracted, skipping analysis")
                                return
                                
                            print("[DEBUG] Extracted terms:", [self._ast_to_str(t, assignment_map) for t in terms])
                            
                            # Limit analysis to prevent excessive computation
                            if len(terms) > 10:
                                print(f"[Term Contribution] Too many terms ({len(terms)}), limiting analysis to first 10")
                                terms = terms[:10]
                            
                            # Analyze both individual terms and term pairs
                            term_contributions, pair_contributions = self._analyze_term_combinations(
                                terms, function_body, return_expr, assignment_map, new_function, avg_score
                            )
                            
                            print("[DEBUG] Term contributions:", term_contributions)
                            print("[DEBUG] Pair contributions:", pair_contributions)
                            
                            # Generate enhanced annotations
                            comment_lines = []
                            
                            # Sort and include only the top/bottom 3 individual term contributions
                            # Apply a small threshold to filter out numerical noise
                            ablation_eps = float(os.environ.get('ABLATION_MIN_DELTA', '1e-6'))
                            valid_term_contribs = [tc for tc in term_contributions if abs(tc[1]) > ablation_eps]
                            positive_terms = sorted([tc for tc in valid_term_contribs if tc[1] > 0], key=lambda x: x[1], reverse=True)[:3]
                            
                            if positive_terms:
                                comment_lines.append("# Top 3 Most Contributing Terms:")
                                for term, contribution in positive_terms:
                                    comment_lines.append(
                                        f"# [Ablation] Removing this term decreases the score by {abs(contribution):.8f}: {term}"
                                    )

                            # Show the three overall least contributions (includes negative and small positive)
                            least_terms_all = sorted(valid_term_contribs, key=lambda x: x[1])
                            least_terms = []
                            for tc in least_terms_all:
                                if tc not in positive_terms:
                                    least_terms.append(tc)
                                if len(least_terms) == 3:
                                    break
                            if least_terms:
                                comment_lines.append("# Top 3 Least Contributing Terms:")
                                for term, contribution in least_terms:
                                    change_word = "increases" if contribution < 0 else "decreases"
                                    comment_lines.append(
                                        f"# [Ablation] Removing this term {change_word} the score by {abs(contribution):.8f}: {term}"
                                    )
                            
                            # Add sorted and limited pair contributions (top/bottom 3)
                            valid_pair_contribs = [pc for pc in pair_contributions if abs(pc[1]) > ablation_eps]
                            positive_pairs = sorted([pc for pc in valid_pair_contribs if pc[1] > 0], key=lambda x: x[1], reverse=True)[:3]
                            least_pairs_all = sorted(valid_pair_contribs, key=lambda x: x[1])
                            least_pairs = []
                            for pc in least_pairs_all:
                                if pc not in positive_pairs:
                                    least_pairs.append(pc)
                                if len(least_pairs) == 3:
                                    break

                            if positive_pairs:
                                comment_lines.append("\n# Top 3 Most Contributing Term Pairs:")
                                for (term1, term2), contribution in positive_pairs:
                                    comment_lines.append(
                                        f"# [Ablation] Removing these terms together decreases the score by {abs(contribution):.8f}:"
                                    )
                                    comment_lines.append(f"#   Term 1: {term1}")
                                    comment_lines.append(f"#   Term 2: {term2}")
                            if least_pairs:
                                comment_lines.append("\n# Top 3 Least Contributing Term Pairs:")
                                for (term1, term2), contribution in least_pairs:
                                    change_word = "increases" if contribution < 0 else "decreases"
                                    comment_lines.append(
                                        f"# [Ablation] Removing these terms together {change_word} the score by {abs(contribution):.8f}:"
                                    )
                                    comment_lines.append(f"#   Term 1: {term1}")
                                    comment_lines.append(f"#   Term 2: {term2}")
                            
                            # Add annotations to function body
                            annotated_body = '\n'.join(comment_lines) + '\n\n' + function_body
                            new_function.body = annotated_body
                            print("[Term Contribution] Annotated function body:")
                            print(annotated_body)
                            
                        except Exception as e:
                            print(f"[WARNING] Term extraction/analysis failed: {e}")
                            print(traceback.format_exc())
                            # Continue without annotations rather than crashing
                            
                    except Exception as e:
                        print(f"[WARNING] Term contribution analysis failed: {e}")
                        print(traceback.format_exc())
                        # Continue without annotations rather than crashing
                    # --- End Term Contribution Analysis ---
                else:
                    print("\nTerm contribution analysis DISABLED - function will be saved without term analysis.")

                # Continue with the rest of the analysis...
                if self._use_atom_analysis and self._atom_analyzer:
                    try:
                        iteration = kwargs.get('global_sample_nums', 0)
                        problem_description = self._extract_problem_description()
                        new_function = self.analyze_with_atom(
                            new_function, 
                            island_id, 
                            iteration, 
                            scores_per_test, 
                            problem_description
                        )
                    except Exception as e:
                        print(f"Failed to run Atom analysis: {e}")
                        print(f"Traceback: {traceback.format_exc()}")
                else:
                    print(f"\nAtom analysis DISABLED - function will be saved without domain knowledge enhancement")
                
                print("\nRegistering successful program with database")
                self._database.register_program(
                    new_function,
                    island_id,
                    scores_per_test,
                    **kwargs,
                    evaluate_time=evaluate_time
                )
                print(f"\n===== FUNCTION SAVED TO ISLAND {island_id} =====")
            else:
                print("\nProgram failed all tests - not registering")
                profiler: profile.Profiler = kwargs.get('profiler', None)
                if profiler:
                    global_sample_nums = kwargs.get('global_sample_nums', None)
                    sample_time = kwargs.get('sample_time', None)
                    new_function.global_sample_nums = global_sample_nums
                    new_function.score = None
                    new_function.sample_time = sample_time
                    new_function.evaluate_time = evaluate_time
                    profiler.register_function(new_function)
                
            print("========== EVALUATOR: Analysis Complete ==========\n")
            
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Analysis failed completely: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print("This sample will be skipped due to critical errors.")
            print("========== EVALUATOR: Analysis Failed ==========\n")

    def analyze_with_atom(self, function, island_id, iteration, scores_per_test, problem_description):
        """Analyze the function with Atom and enhance it with domain knowledge."""
        if not self._use_atom_analysis:
            print("Atom analysis is disabled")
            return function
            
        print("\n----- Running Atom Analysis on Function -----\n")
        
        # Trim problem description for readability
        if problem_description:
            print(f"Problem description extracted: {len(problem_description)} chars")
            print(f"First 100 chars: {problem_description[:100]}...")
            
        print("\nSending function to Atom for analysis...")
        
        # Run analysis with iteration number
        analysis = self._atom_analyzer.analyze_function(
            function, 
            scores_per_test, 
            problem_description,
            iteration=iteration  # Pass the current iteration number
        )
        
        # Enhance the function with domain knowledge
        enhanced_function = self._atom_analyzer.enhance_function_with_analysis(function, analysis)
        
        if enhanced_function != function:
            print("\nSuccessfully enhanced function with Atom analysis")
            print("\n----- Enhanced Function Description -----")
            print(enhanced_function.docstring)
            print("----------------------------------------\n")
            print("\nFunction skeleton enhanced with domain knowledge - ready to save to island", island_id)
        else:
            print("\nNo enhancement was applied to the function (either first iteration or already enhanced)")
            
        return enhanced_function
        
    def report_best_models(self):
        """Report the best models and their scores across all islands."""
        print("\n============== BEST MODEL SCORES ==============")
        
        # Get best scores from each island
        best_function = None
        best_score = float('-inf')
        
        # Collect best scores for each island
        for island_id, score in enumerate(self._database._best_score_per_island):
            print(f"Island {island_id}: {score}")
            
            # Track overall best
            if score > best_score:
                best_score = score
                best_function = self._database._best_program_per_island[island_id]
        
        print("\n-------------- OVERALL BEST MODEL --------------")
        if best_function:
            print(f"Best score: {best_score}")
            print(f"Island: {island_id}")
            print("\n----------- BEST MODEL IMPLEMENTATION -----------")
            print(str(best_function))
            print("------------------------------------------------")
            
            # If Atom was used, also display the domain interpretation
            if self._use_atom_analysis and best_function.docstring and "DOMAIN INTERPRETATION:" in best_function.docstring:
                print("\n----------- DOMAIN INTERPRETATION -----------")
                try:
                    docstring = best_function.docstring
                    start_idx = docstring.find("DOMAIN INTERPRETATION:")
                    if start_idx >= 0:
                        end_idx = docstring.find("SUMMARY:", start_idx)
                        if end_idx >= 0:
                            interpretation = docstring[start_idx:end_idx].strip()
                        else:
                            interpretation = docstring[start_idx:].strip()
                        print(interpretation)
                except Exception as e:
                    print(f"Error extracting domain interpretation: {e}")
                print("--------------------------------------------")
        else:
            print("No successful models found.")
            
        print("===============================================")