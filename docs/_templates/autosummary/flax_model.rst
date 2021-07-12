{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:


   {% block stuff %}{%- if '__init__' in all_methods %}

   .. automethod:: {{ name }}.__init__
   {%- endif -%}{% endblock %}


   {% block methods %}
      {% if attributes %}
   .. rst-class:: class-dl-groups

   Attributes{% block attributes %}{% for item in all_attributes %}{%- if not item.startswith('_') and item not in ['parent', 'scope', 'name'] %}
      .. autoattribute:: {{ name }}.{{ item }}
      {%- endif -%}{%- endfor %}{% endblock %}{% endif %}
   
      {% if methods %}
   .. rst-class:: class-dl-groups

   Methods{% block methodslist %}{% for item in all_methods %}{%- if not item.startswith('_') %}{%- if item not in ['bind', 'clone', 'get_variable', 'has_variable', 'hidden_bias_init', 'param', 'parameter', 'variable', 'sow', 'setup', 'make_rng', 'init', 'apply', 'init_with_output', 'is_mutable_collection'] %}
      .. automethod:: {{ name }}.{{ item }}
      {%- endif -%}{%- endif -%}{%- endfor %}{% endblock %}{% endif %}

   {% endblock %}
