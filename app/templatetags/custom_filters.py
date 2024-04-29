from django import template

register = template.Library()

@register.filter(name='is_dict')
def is_dict(value):
    return isinstance(value, dict)
