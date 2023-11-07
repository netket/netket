.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:


   {% if all_methods is defined and all_methods %}
   {% block init %}{%- if '__init__' in all_methods %}

   .. automethod:: {{ name }}.__init__
   {%- endif -%}{% endblock %}
   {%- endif -%}


   {% block methods %}
      {% if attributes %}
   .. rst-class:: class-dl-groups

   Attributes{% block attributes %}{% for item in all_attributes %}{%- if not item.startswith('_') and item not in ['parent', 'scope', 'name'] and item not in inherited_members %}
      .. autoattribute:: {{ name }}.{{ item }}
      {%- endif -%}{%- endfor %}{% endblock %}{% endif %}
   
      {% if methods %}
   .. rst-class:: class-dl-groups

   Methods{% block methodslist %}
      .. automethod:: __call__
   {% for item in all_methods %}{%- if not item.startswith('_') %}{%- if item not in ['bind', 'setup',
   'apply'] and item not in inherited_members %}
      .. automethod:: {{ name }}.{{ item }}
      {%- endif -%}{%- endif -%}{%- endfor %}{% endblock %}{% endif %}

   {% endblock %}
