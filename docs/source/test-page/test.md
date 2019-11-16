# This is a test page

Right now, there's nothing on it

## Trying out different markdown

### Python code
```python
r = range(10)
for num in r:
    print(r)
```

### Math
Regular:
```math
y = \frac{1}{2} \sin \left(2 \pi t\right)
```

Inline:

This is some inline math: `$\cos\left(2 \pi t\right)$`

### Linking to modules and functions
As far as I can tell, linking directly to a module is only possible by writing the link in rst format rather than markdown. To do so, wrap the paragraph containing the link in an `eval_rst` block. For example,

    ```eval_rst
    This is a link to the documentation for :func:`autodiff.operations.elementary`.
    ```

renders as:
```eval_rst
This is a link to the documentation for :func:`autodiff.operations.elementary`.
```
This is annoying but looks nice.
