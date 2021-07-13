{{ fullname | escape | underline}}

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

   Attributes{% block attributes %}{% for item in all_attributes %}{%- if not item.startswith('_') %}
      .. autoattribute:: {{ name }}.{{ item }}
      {%- endif -%}{%- endfor %}{% endblock %}{% endif %}
   
      {% if methods %}
   .. rst-class:: class-dl-groups

   Methods{% block methodslist %}{% for item in all_methods %}{%- if not item.startswith('_') or item in ['__call__'] %}
      .. automethod:: {{ name }}.{{ item }}
      {%- endif -%}{%- endfor %}{% endblock %}{% endif %}

   {% endblock %}
