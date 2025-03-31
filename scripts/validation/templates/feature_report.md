# Feature Analysis Report: {{ feature_name }}

**Feature Type:** {{ feature_type|title }}  
**Generated at:** {{ timestamp }}

## Basic Statistics

{{ basic_stats|replace('<table>', '')|replace('</table>', '')|replace('<thead>', '')|replace('</thead>', '')|replace('<tbody>', '')|replace('</tbody>', '') }}

## Statistical Test Results

{% for line in statistical_results.split('\n') %}
{{ line }}
{% endfor %}

## Visualizations

{% for viz_type, viz_path in visualization_paths.items() %}
### {{ viz_type|replace('_', ' ')|title }}

![{{ viz_type }} visualization]({{ viz_path }})

{% endfor %} 