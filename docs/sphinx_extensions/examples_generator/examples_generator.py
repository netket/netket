#!/usr/bin/env python3
"""
Sphinx extension providing directives for NetKet examples documentation.

This extension scans the Examples/ directory and provides MyST directives
for generating dynamic content about examples in documentation pages.
Available directives:
- toctree_examples: Generate a toctree with example categories
- category_cards: Generate grid cards for each category
- quick_reference: Generate a reference table
- list_examples: List examples with filtering options
"""

import ast
import pathlib
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from sphinx.application import Sphinx
from sphinx.util import logging
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective
from docutils import nodes

logger = logging.getLogger(__name__)

@dataclass
class ExampleInfo:
    """Information about a single example file."""
    name: str
    path: str
    category: str
    title: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    github_url: str = ""
    tags: List[str] = field(default_factory=list)

class ExamplesGenerator:
    """Generator for NetKet examples documentation."""
    
    def __init__(self, examples_dir: str, docs_dir: str, base_github_url: str = "https://github.com/netket/netket"):
        self.examples_dir = pathlib.Path(examples_dir)
        self.docs_dir = pathlib.Path(docs_dir)
        self.base_github_url = base_github_url
        self.templates_dir = pathlib.Path(__file__).parent
        
    def extract_file_metadata(self, filepath: pathlib.Path) -> ExampleInfo:
        """Extract metadata from a Python file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {filepath}: {e}")
            return self._create_default_example_info(filepath)
        
        # Parse the file
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Could not parse {filepath}: {e}")
            return self._create_default_example_info(filepath)
            
        # Extract information
        info = self._create_default_example_info(filepath)
        
        # Extract title and description from docstring or comments
        info.title, info.description = self._extract_title_and_description(content, tree)
        
        # If no title found, use filename as title
        if not info.title:
            info.title = filepath.stem.replace('_', ' ').title()
        
        # Set GitHub URL
        rel_path = filepath.relative_to(self.examples_dir.parent)
        info.github_url = f"{self.base_github_url}/blob/master/{rel_path}"
        
        # Generate tags based on filename and content
        info.tags = self._generate_tags(filepath, content)
        
        return info
    
    def _create_default_example_info(self, filepath: pathlib.Path) -> ExampleInfo:
        """Create a default ExampleInfo object."""
        category = filepath.parent.name
        name = filepath.stem
        
        return ExampleInfo(
            name=name,
            path=str(filepath),
            category=category,
            title="",
            description="",
            github_url="",
            tags=[]
        )
    
    def _extract_title_and_description(self, content: str, tree: ast.AST) -> Tuple[str, str]:
        """Extract title and description from docstring or comments."""
        title = ""
        description = ""
        
        # Try to get module docstring (handle both old and new Python AST)
        if (tree.body and 
            isinstance(tree.body[0], ast.Expr)):
            # Handle both ast.Str (Python < 3.8) and ast.Constant (Python >= 3.8)
            if isinstance(tree.body[0].value, ast.Str):
                docstring = tree.body[0].value.s.strip()
            elif isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
                docstring = tree.body[0].value.value.strip()
            else:
                docstring = None
                
            if docstring:
                lines = docstring.split('\n')
                if lines:
                    title = lines[0].strip()
                    if len(lines) > 1:
                        # Find the next non-empty line as description
                        for line in lines[1:]:
                            line = line.strip()
                            if line and not line.startswith('Tags:'):
                                description = line
                                break
                return title, description
        
        # Look for comment blocks at the beginning
        lines = content.split('\n')
        description_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip copyright blocks but continue processing other comments
            if 'Copyright' in line:
                continue
                
            # Look for descriptive comments
            if line.startswith('#') and not line.startswith('#!/'):
                desc_line = line.lstrip('#').strip()
                if desc_line:
                    description_lines.append(desc_line)
            elif line and not line.startswith('#') and not line.startswith('import'):
                # Stop at first actual code line
                break
        
        # If found description lines, use first as title, rest as description
        if description_lines:
            title = description_lines[0]
            if len(description_lines) > 1:
                description = ' '.join(description_lines[1:])
        
        return title, description
    
    
    
    def _generate_tags(self, filepath: pathlib.Path, content: str) -> List[str]:
        """Generate tags based on filename and content."""
        tags = []
        
        # Add category as tag
        tags.append(filepath.parent.name.lower())
        
        # Common keywords to look for in content
        keywords = {
            'ising': ['ising', 'Ising'],
            'heisenberg': ['heisenberg', 'Heisenberg'],
            'hubbard': ['hubbard', 'Hubbard'],
            'fermion': ['fermion', 'Fermion'],
            'boson': ['boson', 'Boson'],
            'rbm': ['RBM', 'rbm'],
            'autoreg': ['autoreg', 'autoregressive'],
            'jastrow': ['jastrow', 'Jastrow'],
            'gcnn': ['GCNN', 'gcnn'],
            'vmc': ['VMC', 'vmc'],
            'mcmc': ['MCMC', 'mcmc'],
            'ground_state': ['ground', 'Ground'],
            'dynamics': ['dynamics', 'Dynamics', 'time_evolution'],
            'continuous': ['continuous', 'Continuous'],
            'discrete': ['discrete', 'Discrete'],
            '1d': ['1d', '1D'],
            '2d': ['2d', '2D'],
            'lattice': ['lattice', 'Lattice'],
            'graph': ['graph', 'Graph'],
        }
        
        for tag, patterns in keywords.items():
            for pattern in patterns:
                if pattern in content or pattern in filepath.name:
                    tags.append(tag)
                    break
        
        return list(set(tags))
    
    def _parse_readme_content(self, category_dir: pathlib.Path) -> Dict[str, str]:
        """Parse content directly from README.md file in the category directory."""
        readme_path = category_dir / "README.md"
        if not readme_path.exists():
            return {}
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {readme_path}: {e}")
            return {}
        
        # Remove YAML frontmatter if present
        frontmatter_pattern = r'---\s*\n.*?\n---\s*\n'
        content = re.sub(frontmatter_pattern, '', content, flags=re.DOTALL)
        
        # Remove all # headings (titles)
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            if line.strip().startswith('# '):
                continue
            processed_lines.append(line)
        
        # Join back and strip leading/trailing whitespace
        content = '\n'.join(processed_lines).strip()
        
        # Extract short description (first line before double newline)
        short_desc = ""
        full_desc = content
        
        # Find first paragraph (content before first double newline)
        if '\n\n' in content:
            short_desc = content.split('\n\n')[0].strip()
        elif content:
            # If no double newline, use first line
            first_line = content.split('\n')[0].strip()
            if first_line:
                short_desc = first_line
        
        return {
            'short_description': short_desc,
            'description': full_desc
        }
    
    
    def scan_examples(self) -> Dict[str, List[ExampleInfo]]:
        """Scan the examples directory and return organized examples."""
        examples_by_category = {}
        
        if not self.examples_dir.exists():
            logger.error(f"Examples directory not found: {self.examples_dir}")
            return examples_by_category
        
        # Scan all subdirectories
        for category_dir in self.examples_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue
                
            category = category_dir.name
            examples = []
            
            # Find all Python files in this category
            for py_file in category_dir.glob('*.py'):
                if py_file.name.startswith('_') or 'test' in py_file.name.lower():
                    continue
                    
                example_info = self.extract_file_metadata(py_file)
                examples.append(example_info)
            
            if examples:
                # Sort examples by name
                examples.sort(key=lambda x: x.name)
                examples_by_category[category] = examples
        
        return examples_by_category
    
    def _get_category_description(self, category: str) -> Dict[str, str]:
        """Get a detailed description for each category from README.md files."""
        # Try to read from README.md file first
        category_dir = self.examples_dir / category
        if category_dir.exists():
            readme_content = self._parse_readme_content(category_dir)
            if readme_content and readme_content.get('description'):
                return {
                    'desc': readme_content['short_description'] or f'Examples related to {category.replace("_", " ").lower()}.',
                    'physics': 'Quantum many-body systems',  # Default fallback
                    'techniques': 'Neural quantum states, variational methods'  # Default fallback
                }
        
        # Fallback for special categories that don't have directories (like Heisenberg1d, HeisenbergJ1J2)
        fallback_descriptions = {
            'Heisenberg1d': {
                'desc': 'One-dimensional Heisenberg model implementations. A canonical example of quantum many-body physics with exact solutions for comparison.',
                'physics': '1D quantum magnetism, Bethe ansatz, spin chains',
                'techniques': 'Variational Monte Carlo, exact benchmarks'
            },
            'HeisenbergJ1J2': {
                'desc': 'J1-J2 Heisenberg model with competing interactions. This model exhibits rich phase diagrams including quantum spin liquid phases.',
                'physics': 'Frustrated magnetism, quantum spin liquids, phase transitions',
                'techniques': 'Competing interactions, phase diagram studies'
            }
        }
        
        return fallback_descriptions.get(category, {
            'desc': f'Examples related to {category.replace("_", " ").lower()}.',
            'physics': 'Quantum many-body systems',
            'techniques': 'Neural quantum states, variational methods'
        })
    
    
    def _load_template(self, template_name: str) -> str:
        """Load a template file."""
        template_path = self.templates_dir / template_name
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not load template {template_name}: {e}")
            return ""
    
    def generate_category_page(self, category: str, examples: List[ExampleInfo]) -> str:
        """Generate a documentation page for a category."""
        # Create a nice category title
        title = category.replace('_', ' ').replace('-', ' ').title()
        
        # Get category metadata and README content
        cat_info = self._get_category_description(category)
        category_dir = self.examples_dir / category
        readme_content = self._parse_readme_content(category_dir) if category_dir.exists() else {}
        
        # Use full description from README if available, otherwise use short description
        full_description = readme_content.get('description', cat_info['desc']) if readme_content else cat_info['desc']
        
        # Generate examples content
        examples_content = ""
        example_template = self._load_template("example_entry.md")
        
        for i, example in enumerate(examples, 1):
            display_name = example.title if example.title else example.name
            
            # Prepare tags line
            tags_line = ""
            if example.tags:
                tags_str = " | ".join(f"`{tag}`" for tag in example.tags)
                tags_line = f"{tags_str}\n"
            
            # Prepare description
            description = example.description if example.description else ""
            
            # Format the example entry
            if example_template:
                example_entry = example_template.format(
                    index=i,
                    display_name=display_name,
                    github_url=example.github_url,
                    tags_line=tags_line,
                    description=description
                )
            else:
                # Fallback if template loading fails
                example_entry = f"### {i}. [{display_name}]({example.github_url})\n\n{tags_line}\n{description}\n\n"
            
            examples_content += example_entry
        
        # Load and format the main template
        category_template = self._load_template("category_page.md")
        if category_template:
            content = category_template.format(
                title=title,
                description=full_description,
                physics=cat_info['physics'],
                techniques=cat_info['techniques'],
                num_examples=len(examples),
                examples_content=examples_content
            )
        else:
            # Fallback if template loading fails
            content = f"# {title} Examples\n\n{full_description}\n\n{examples_content}"
                
        return content
        
    
    def generate_examples_list(self, category: Optional[str] = None, tags: Optional[List[str]] = None, limit: Optional[int] = None) -> str:
        """Generate a markdown list of examples for the list_examples directive."""
        examples_by_category = self.scan_examples()
        
        if not examples_by_category:
            return "No examples found."
        
        # Filter examples based on criteria
        filtered_examples = []
        
        for cat, examples in examples_by_category.items():
            if category and cat != category:
                continue
                
            for example in examples:
                if tags:
                    if not any(tag in example.tags for tag in tags):
                        continue
                        
                filtered_examples.append((cat, example))
        
        # Sort by category then by name
        filtered_examples.sort(key=lambda x: (x[0], x[1].name))
        
        # Apply limit if specified
        if limit:
            filtered_examples = filtered_examples[:limit]
        
        # Generate markdown list
        content = ""
        current_category = None
        
        for cat, example in filtered_examples:
            if current_category != cat:
                current_category = cat
                category_title = cat.replace('_', ' ').replace('-', ' ').title()
                content += f"\n### {category_title}\n\n"
            
            display_name = example.title if example.title else example.name
            content += f"- **[{display_name}]({example.github_url})**"
            if example.description:
                content += f": {example.description}"
            
            if example.tags:
                tags_str = ", ".join(f"`{tag}`" for tag in example.tags[:3])
                content += f" | Tags: {tags_str}"
            
            content += "\n"
        
        return content
    

# Global reference to the generator for directive access
_examples_generator = None

class ListExamplesDirective(SphinxDirective):
    """Directive to list examples in markdown files."""
    
    has_content = False
    option_spec = {
        'category': directives.unchanged,
        'tags': directives.unchanged,
        'limit': directives.positive_int,
    }
    
    def run(self):
        global _examples_generator
        
        if _examples_generator is None:
            return [nodes.paragraph(text="Examples generator not initialized.")]
        
        # Parse options
        category = self.options.get('category')
        tags = self.options.get('tags')
        if tags:
            tags = [tag.strip() for tag in tags.split(',')]
        limit = self.options.get('limit')
        
        # Generate examples list
        examples_content = _examples_generator.generate_examples_list(
            category=category, 
            tags=tags, 
            limit=limit
        )
        
        # Create a raw node with the markdown content
        raw_node = nodes.raw('', examples_content, format='markdown')
        return [raw_node]


class ToctreeExamplesDirective(SphinxDirective):
    """Directive to generate a toctree for examples."""
    
    has_content = False
    option_spec = {
        'maxdepth': directives.positive_int,
        'caption': directives.unchanged,
        'hidden': directives.flag,
    }
    
    def run(self):
        global _examples_generator
        
        if _examples_generator is None:
            return [nodes.paragraph(text="Examples generator not initialized.")]
        
        examples_by_category = _examples_generator.scan_examples()
        
        # Generate toctree content
        maxdepth = self.options.get('maxdepth', 1)
        caption = self.options.get('caption', 'Example Categories')
        hidden = 'hidden' in self.options
        
        toctree_content = f"```{{toctree}}\n:maxdepth: {maxdepth}\n:caption: {caption}\n"
        if hidden:
            toctree_content += ":hidden:\n"
        toctree_content += "\n"
        
        for category in sorted(examples_by_category.keys()):
            toctree_content += f"{category}/index\n"
        
        toctree_content += "```"
        
        # Create a raw node with the markdown content
        raw_node = nodes.raw('', toctree_content, format='markdown')
        return [raw_node]


class CategoryCardsDirective(SphinxDirective):
    """Directive to generate category cards for examples."""
    
    has_content = False
    option_spec = {
        'columns': directives.positive_int,
        'gutter': directives.positive_int,
    }
    
    def run(self):
        global _examples_generator
        
        if _examples_generator is None:
            return [nodes.paragraph(text="Examples generator not initialized.")]
        
        examples_by_category = _examples_generator.scan_examples()
        
        # Load templates
        card_template = _examples_generator._load_template("category_card.html")
        wrapper_template = _examples_generator._load_template("category_cards_wrapper.html")
        
        # Sort categories by popularity
        category_order = ['Ising', 'Heisenberg1d', 'FullSummation', 'Heisenberg', 'SR', 'Continuous', 'nn_frameworks', 'Autoregressive', 'Fermions', 'Dynamics', 'HeisenbergJ1J2', 'DissipativeIsing1d', 'StateReconstruction', 'Sharding']
        sorted_categories = sorted(examples_by_category.keys(), key=lambda x: category_order.index(x) if x in category_order else 999)
        
        cards_content = ""
        for category in sorted_categories:
            examples = examples_by_category[category]
            category_title = category.replace('_', ' ').replace('-', ' ').title()
            cat_info = _examples_generator._get_category_description(category)
            
            # Create card HTML
            desc_short = cat_info['desc'][:120] + ('...' if len(cat_info['desc']) > 120 else '')
            physics_short = cat_info['physics'][:50] + ('...' if len(cat_info['physics']) > 50 else '')
            
            if card_template:
                card_html = card_template.format(
                    category=category,
                    category_title=category_title,
                    desc_short=desc_short,
                    num_examples=len(examples),
                    physics_short=physics_short
                )
            else:
                # Fallback
                card_html = f'<div class="card"><h5>{category_title}</h5><p>{desc_short}</p></div>'
            
            cards_content += card_html
        
        # Wrap in container
        if wrapper_template:
            html_content = wrapper_template.format(cards_content=cards_content)
        else:
            html_content = f'<div class="container-fluid"><div class="row">{cards_content}</div></div>'
        
        # Return as raw HTML node
        return [nodes.raw('', html_content, format='html')]


class QuickReferenceDirective(SphinxDirective):
    """Directive to generate quick reference table for examples."""
    
    has_content = False
    option_spec = {}
    
    def run(self):
        global _examples_generator
        
        if _examples_generator is None:
            return [nodes.paragraph(text="Examples generator not initialized.")]
        
        examples_by_category = _examples_generator.scan_examples()
        
        # Load templates
        table_template = _examples_generator._load_template("quick_reference_table.html")
        row_template = _examples_generator._load_template("table_row.html")
        
        # Sort categories by popularity
        category_order = ['Ising', 'Heisenberg1d', 'FullSummation', 'Heisenberg', 'SR', 'Continuous', 'nn_frameworks', 'Autoregressive', 'Fermions', 'Dynamics', 'HeisenbergJ1J2', 'DissipativeIsing1d', 'StateReconstruction', 'Sharding']
        sorted_categories = sorted(examples_by_category.keys(), key=lambda x: category_order.index(x) if x in category_order else 999)
        
        table_rows = ""
        for category in sorted_categories:
            examples = examples_by_category[category]
            category_title = category.replace('_', ' ').replace('-', ' ').title()
            cat_info = _examples_generator._get_category_description(category)
            
            physics_text = cat_info['physics'][:30] + ('...' if len(cat_info['physics']) > 30 else '')
            techniques_text = cat_info['techniques'][:35] + ('...' if len(cat_info['techniques']) > 35 else '')
            
            if row_template:
                row_html = row_template.format(
                    category=category,
                    category_title=category_title,
                    num_examples=len(examples),
                    physics_text=physics_text,
                    techniques_text=techniques_text
                )
            else:
                # Fallback
                row_html = f'<tr><td>{category_title}</td><td>{len(examples)}</td><td>{physics_text}</td><td>{techniques_text}</td></tr>\n'
            
            table_rows += row_html
        
        if table_template:
            content = table_template.format(table_rows=table_rows)
        else:
            content = f'<table class="table table-striped"><thead><tr><th>Category</th><th>Examples</th><th>Key Physics</th><th>Primary Techniques</th></tr></thead><tbody>{table_rows}</tbody></table>'
        
        # Create a raw node with the HTML content
        raw_node = nodes.raw('', content, format='html')
        return [raw_node]

def setup(app: Sphinx) -> Dict[str, any]:
    """Setup the Sphinx extension."""
    global _examples_generator
    
    def init_examples_generator(app: Sphinx) -> None:
        """Initialize the examples generator during the build process."""
        global _examples_generator
        
        # Get paths relative to the conf.py file
        conf_dir = pathlib.Path(app.confdir)
        examples_dir = conf_dir.parent / "Examples"
        docs_dir = conf_dir  # Generate in docs source directory for toctree inclusion
        
        _examples_generator = ExamplesGenerator(str(examples_dir), str(docs_dir))
        
        # Only generate category pages, not the main index
        examples_by_category = _examples_generator.scan_examples()
        
        if examples_by_category:
            # Create examples directory
            examples_doc_dir = _examples_generator.docs_dir / "examples"
            examples_doc_dir.mkdir(exist_ok=True)
            
            # Generate category pages only
            for category, examples in examples_by_category.items():
                category_dir = examples_doc_dir / category
                category_dir.mkdir(exist_ok=True)
                
                category_content = _examples_generator.generate_category_page(category, examples)
                with open(category_dir / "index.md", 'w', encoding='utf-8') as f:
                    f.write(category_content)
            
            logger.info(f"Generated documentation for {len(examples_by_category)} example categories")
    
    # Register all the directives
    app.add_directive('list_examples', ListExamplesDirective)
    app.add_directive('toctree_examples', ToctreeExamplesDirective)
    app.add_directive('category_cards', CategoryCardsDirective)
    app.add_directive('quick_reference', QuickReferenceDirective)
    
    # Connect to the build process
    app.connect('builder-inited', init_examples_generator)
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }