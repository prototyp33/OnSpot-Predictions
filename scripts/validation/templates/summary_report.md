# Distribution Analysis Summary Report

**Generated at:** {{ timestamp }}

## Overview

- **Total Features:** {{ total_features }}
- **Features with Issues:** {{ features_with_issues }}
- **Features with Warnings:** {{ features_with_warnings }}

{% if issues %}
## Critical Issues

{% for issue in issues %}
### {{ issue.feature }}
- **Type:** {{ issue.type|replace('_', ' ')|title }}
- **Details:** {{ issue.details }}

{% endfor %}
{% endif %}

{% if warnings %}
## Warnings

{% for warning in warnings %}
### {{ warning.feature }}
- **Type:** {{ warning.type|replace('_', ' ')|title }}
- **Details:** {{ warning.details }}

{% endfor %}
{% endif %} 