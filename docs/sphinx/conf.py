"""Sphinx configuration for the Maestro Python documentation.

Build from the repository root:

    pip install -r docs/sphinx/requirements.txt
    sphinx-build -W --keep-going -b html docs/sphinx docs/sphinx/_build/html
    sphinx-build -b linkcheck docs/sphinx docs/sphinx/_build/linkcheck

``maestro`` must be importable, either from an installed wheel or from a local
build on ``PYTHONPATH``. The API reference and the executed pages run against it.
"""

import importlib.util
import pathlib
import re
import tomllib

from docutils import nodes
from sphinx import addnodes

ROOT = pathlib.Path(__file__).resolve().parents[2]

project = "Maestro"
author = "Qoro Quantum"
copyright = "Qoro Quantum"
release = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_reredirects",
]

exclude_patterns = ["_build", "jupyter_execute", "requirements.txt"]

# MyST and notebook execution. Pages with code cells run at build time, so a
# broken example fails the build instead of shipping.
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 300
# The Deltakit page runs deltakit-maestro, which requirements.txt installs only
# where it has a release for the Python that builds the docs. Without it the
# page is rendered with its code and not run.
if importlib.util.find_spec("deltakit_maestro") is None:
    nb_execution_excludepatterns = ["*getting-started/deltakit.md"]

# API reference.
autodoc_default_options = {"members": True, "undoc-members": True}
autodoc_typehints = "signature"
add_module_names = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Theme. Matches the Divi docs: Furo with Qoro's brand colours and fonts.
html_theme = "furo"
html_title = f"Maestro {release}"
html_favicon = str(ROOT / "markdown" / "favicon.ico")
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#CC3366",
        "color-brand-content": "#CC3366",
        "font-stack": "Poppins, sans-serif",
        "font-stack--headings": "Poppins, sans-serif",
        "font-stack--monospace": "ui-monospace, 'Cascadia Code', 'Source Code Pro', Menlo, Consolas, 'Liberation Mono', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FDB9FD",
        "color-brand-content": "#FDB9FD",
        "color-background-primary": "#131313",
        "color-background-secondary": "#1a1a1a",
        "color-foreground-primary": "#E9EEF6",
    },
}
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# The Python guide used to live in the Doxygen site at these paths. The C++
# pages keep their paths under /cpp/, and the deploy workflow redirects their
# old root URLs there.
redirects = {
    "d7/d01/python_guide": "/user-guide/index.html",
    "d2/d02/py_quickstart": "/getting-started/first-circuit.html",
    "d2/d00/py_circuits": "/user-guide/building-circuits.html",
    "dc/d03/py_backends": "/user-guide/choosing-a-method.html",
    "d4/d01/py_algorithms": "/user-guide/expectation-values.html",
    "d5/d03/py_noise": "/user-guide/noise-and-qec.html",
    "d6/d02/py_sinter": "/user-guide/qec-sinter.html",
    "dc/d01/py_hpc": "/getting-started/gpu-access.html",
    "dc/d01/py_reference": "/api/index.html",
}

# Link checking.
linkcheck_ignore = [
    # Published together with this site by .github/workflows/docs.yml.
    r"https://qoroquantum\.github\.io/maestro/cpp/",
]
# GitHub renders README anchors client-side.
linkcheck_anchors_ignore_for_url = [r"https://github\.com/"]
linkcheck_retries = 2
linkcheck_timeout = 30

# nanobind reports types from the internal extension module (maestro.maestro.*)
# and prints SimulatorConfig defaults as their full repr. Rewrite both so
# signatures show the public names and stay readable.
_INTERNAL_MODULE = re.compile(r"\bmaestro\.maestro\.")
_CONFIG_REPR = re.compile(r"SimulatorConfig\([^()]*\)")
_EXAMPLE_HEADER_RE = re.compile(r"^(Example|Examples)(?:\s*\((.*?)\))?:\s*(.*)$")


def _clean(text):
    if not text:
        return text
    return _CONFIG_REPR.sub("SimulatorConfig()", _INTERNAL_MODULE.sub("maestro.", text))


def _format_examples(lines):
    new_lines = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        m = _EXAMPLE_HEADER_RE.match(stripped)
        if m:
            heading, note, rest = m.groups()
            rest = rest.strip()
            indent = line[:len(line) - len(line.lstrip())]

            # Skip prose descriptions starting with 'Example:' (e.g. 'qc.prob(...) returns...')
            if rest and " returns " in rest:
                new_lines.append(line)
                i += 1
                continue

            if new_lines and new_lines[-1].strip():
                new_lines.append("")

            new_lines.append(f"{indent}Example:")
            new_lines.append(f"{indent}    .. code-block:: python")
            new_lines.append("")
            if note:
                new_lines.append(f"{indent}        # {note}")
            if rest:
                new_lines.append(f"{indent}        {rest}")
                i += 1
                continue

            i += 1
            while i < n:
                curr = lines[i]
                if not curr.strip():
                    new_lines.append("")
                    i += 1
                    continue
                if curr.startswith(" ") or curr.startswith("\t"):
                    new_lines.append(f"{indent}    {curr}")
                    i += 1
                else:
                    break
            continue

        new_lines.append(line)
        i += 1
    return new_lines


def _clean_docstring(app, what, name, obj, options, lines):
    cleaned = [_clean(line) for line in lines]
    lines[:] = _format_examples(cleaned)


def _clean_signatures(app, doctree):
    # autodoc-process-signature only sees the first overload, so rewrite the
    # rendered signatures instead, including the targets of type cross-references.
    for signature in doctree.findall(addnodes.desc_signature):
        for xref in signature.findall(addnodes.pending_xref):
            for key, value in xref.attributes.items():
                if isinstance(value, str):
                    xref[key] = _clean(value)
        for text in list(signature.findall(nodes.Text)):
            if (cleaned := _clean(str(text))) != str(text):
                text.parent.replace(text, nodes.Text(cleaned))


def _drop_internal_aliases(app, env):
    # autodoc also registers each object under its internal module path, and
    # references resolved from inside ``maestro`` would otherwise pick that name.
    objects = env.get_domain("py").objects
    for name in [name for name in objects if _INTERNAL_MODULE.match(name)]:
        del objects[name]


def setup(app):
    # Connect with lower priority than default (500) so Napoleon (at 500) runs
    # after _clean_docstring has formatted Examples as code blocks.
    app.connect("autodoc-process-docstring", _clean_docstring, priority=400)
    app.connect("doctree-read", _clean_signatures)
    app.connect("env-updated", _drop_internal_aliases)
