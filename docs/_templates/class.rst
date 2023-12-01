{{ cls_info.cls_name }}
{{ "=" * cls_info.cls_name|length }}

.. class:: {{ init_method.name }}{{ init_method.signature }}

{{ cls_info.description }}

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - Parameters
     - {{ init_method.description }}

   * - Attributes
     - 
       {% for prop in properties %}
         {{ prop.description }}
       {% endfor %}

.. raw:: html

   <h2>Example</h2>

{{ cls_info.example }}

.. raw:: html

   <h2>Methods</h2>

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Method
     - Description
   {% for method in methods %}
   * - :meth:`~{{ cls_info.full_cls_name }}.{{ method.name }}`
     - {{ method.short_description }}
   {% endfor %}


{% for method in methods %}
.. automethod:: {{ cls_info.full_cls_name }}.{{ method.name }}
{% endfor %}
