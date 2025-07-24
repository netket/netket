"""
Sphinx extension for NetKet citation system.

This extension provides directives to display citation information from NetKet's
citation system.
"""

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util import logging
import importlib.util
import sys
from pathlib import Path


logger = logging.getLogger(__name__)


def _extract_doi_from_bibtex(bibtex_entry):
    """Extract DOI from bibtex entry."""
    import re
    
    # Pattern to match doi field
    doi_patterns = [
        r'doi\s*=\s*{([^}]+)}',
        r'doi\s*=\s*"([^"]+)"',
        r'doi\s*=\s*([^,\s}]+)',
    ]
    
    for pattern in doi_patterns:
        match = re.search(pattern, bibtex_entry, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def _extract_url_from_bibtex(bibtex_entry):
    """Extract URL from bibtex entry."""
    import re
    
    # Pattern to match url field
    url_patterns = [
        r'url\s*=\s*{([^}]+)}',
        r'url\s*=\s*"([^"]+)"',
        r'url\s*=\s*([^,\s}]+)',
    ]
    
    for pattern in url_patterns:
        match = re.search(pattern, bibtex_entry, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def _extract_title_from_bibtex(bibtex_entry):
    """Extract and clean title from bibtex entry."""
    import re
    
    # Pattern to match title field
    title_patterns = [
        r'title\s*=\s*{([^}]+)}',
        r'title\s*=\s*"([^"]+)"',
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, bibtex_entry, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Remove LaTeX formatting
            title = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', title)
            title = re.sub(r'[{}]', '', title)
            return title
    
    return None


def _generate_bootstrap_citations():
    """Generate bootstrap card-based citation HTML."""
    from netket.utils.citations import get_all_citations
    from netket.utils.citations.core import register_citation
    from netket import __version__ as netket_version
    import uuid
    
    # Get all citations
    citation_registry = get_all_citations()
    
    if not citation_registry:
        return '<div class="alert alert-info">No citations have been registered.</div>'
    
    # Filter out NetKet papers
    netket_papers = ["netket3:2022", "netket2:2019", "netket3:2021"]
    filtered_registry = {k: v for k, v in citation_registry.items() if k not in netket_papers}
    
    if not filtered_registry:
        return '<div class="alert alert-info">No algorithm-specific citations have been registered.</div>'
    
    # Group citations by group_id
    groups = {}
    for key, entry in filtered_registry.items():
        group_id = entry.get('group_id')
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append((key, entry))
    
    html_parts = ['<div class="row">']
    
    for group_id, group_entries in groups.items():
        # Get the first entry to extract group-level info
        first_key, first_entry = group_entries[0]
        condition = first_entry.get('condition', '').strip()
        message = first_entry.get('message', 'This work used the methods described in Ref.~').strip()
        
        # Create bootstrap card
        html_parts.append('<div class="col-12 mb-2">')
        html_parts.append('<div class="card">')
        html_parts.append('<div class="card-body p-2">')
        
        # Combine condition, message and references in one paragraph
        refs_html = []
        for ref_key, entry in group_entries:
            bibtex = entry.get('bibtex', '')
            if bibtex:
                title = _extract_title_from_bibtex(bibtex)
                doi = _extract_doi_from_bibtex(bibtex)
                url = _extract_url_from_bibtex(bibtex)
                
                # Create link
                if doi:
                    link_url = f"https://doi.org/{doi}"
                elif url:
                    link_url = url
                else:
                    link_url = None
                
                if link_url and title:
                    refs_html.append(f'(<code>{ref_key}</code>)<a href="{link_url}" target="_blank">"{title}"</a>')
                elif title:
                    refs_html.append(f'(<code>{ref_key}</code>)"{title}"')
                else:
                    refs_html.append(f'<code>{ref_key}</code>')
            else:
                refs_html.append(f'<code>{ref_key}</code>')
        
        # Create single paragraph with condition (if exists), message, and references
        html_parts.append('<p class="card-text mb-1">')
        if condition:
            html_parts.append(f'<strong>{condition}:</strong> ')
        html_parts.append(f'<em>{message}</em>')
        if refs_html:
            html_parts.append(' <small class="text-muted">')
            html_parts.append(' '.join(refs_html))
            html_parts.append('</small>')
        html_parts.append('</p>')
        
        # Add collapsible BibTeX entries
        bibtex_entries = [entry.get('bibtex', '') for _, entry in group_entries if entry.get('bibtex')]
        if bibtex_entries:
            html_parts.append('<details>')
            html_parts.append('<summary><small>BibTeX entries</small></summary>')
            html_parts.append('<pre class="mb-0"><code class="language-bibtex">')
            html_parts.append('\n\n'.join(bibtex_entries))
            html_parts.append('</code></pre>')
            html_parts.append('</details>')
        
        html_parts.append('</div>')  # card-body
        html_parts.append('</div>')  # card
        html_parts.append('</div>')  # col
    
    html_parts.append('</div>')  # row
    
    return ''.join(html_parts)


class CitationsListNode(nodes.Element):
    """Custom node for citations list."""
    pass


class CitationsListDirective(SphinxDirective):
    """
    Directive to display all citations from NetKet's citation system.
    
    Example:
        .. netket-citations::
           :format: detailed
    """
    
    has_content = False
    optional_arguments = 0
    option_spec = {
        'format': str,  # 'detailed' or 'simple'
    }
    
    def run(self):
        node = CitationsListNode()
        node['format'] = self.options.get('format', 'detailed')
        return [node]


def visit_citations_list_node(self, node):
    """Visit function for CitationsListNode."""
    format_type = node['format']
    
    try:
        # Import NetKet to load citation registrations
        import netket as nk
        from netket.utils.citations import get_all_citations
        import re
        import uuid
        
        if format_type == 'detailed':
            citations_html = _generate_bootstrap_citations()
        else:
            from netket.utils.citations.docs import get_citations_for_docs
            citations_html = get_citations_for_docs(format_type)
        
        self.body.append(citations_html)
        
    except ImportError as e:
        logger.warning(f"Could not import NetKet for citations: {e}")
        self.body.append('<div class="admonition warning"><p class="admonition-title">Warning</p>')
        self.body.append('<p>Citations could not be loaded. NetKet import failed.</p></div>')
    except Exception as e:
        logger.warning(f"Error generating citations: {e}")
        self.body.append('<div class="admonition error"><p class="admonition-title">Error</p>')
        self.body.append(f'<p>Error generating citations: {e}</p></div>')


def depart_citations_list_node(self, node):
    """Depart function for CitationsListNode."""
    pass


def setup(app: Sphinx):
    """Setup function for the Sphinx extension."""
    
    # Add the custom directive
    app.add_directive('netket-citations', CitationsListDirective)
    
    # Add the custom node and its handlers
    app.add_node(
        CitationsListNode,
        html=(visit_citations_list_node, depart_citations_list_node),
        latex=(visit_citations_list_node, depart_citations_list_node),
        text=(visit_citations_list_node, depart_citations_list_node),
    )
    
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }