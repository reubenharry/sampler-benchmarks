# Grid Search Iteration Behavior

## Overview
Both L and step size grid searches now follow a consistent pattern:
- **2 grid iterations by default** (configurable via `grid_iterations` parameter)
- **Boundary expansion**: When a boundary is hit, that iteration is repeated with expanded range (up to 4 total attempts)

## Grid Iteration Structure

### 1. Grid Iterations (Default: 2)
Each grid search performs 2 iterations by default:

**Iteration 1 (Initial Search):**
- L: Search around initial_L with range `[initial_L * 0.5, initial_L * 2.0]`
- Step size: Search around adapted step size within L/step_size constraints

**Iteration 2 (Refinement):**
- L: Refine around the optimal value from iteration 1
- Step size: Refine around the optimal value from iteration 1

### 2. Boundary Expansion (Up to 4 attempts per iteration)
When a boundary is hit during any iteration, that iteration is repeated with expanded range:

**Attempt 1:** Original range
**Attempt 2:** Range expanded by factor 3
**Attempt 3:** Range expanded by factor 9  
**Attempt 4:** Range expanded by factor 27

If still hitting boundary after 4 attempts, an error is raised.

## Implementation Details

### L Grid Search
```python
for grid_iteration in range(grid_iterations):  # Default: 2 iterations
    # Set up L range for this iteration
    if grid_iteration == 0:
        L_range = optimal_L * jnp.linspace(0.5, 2.0, grid_size)
    else:
        # Refine around previous optimal value
        L_range = jnp.linspace(L_min, L_max, grid_size)
    
    # Boundary expansion loop (up to 4 attempts)
    expansion_iteration = 0
    max_expansions = 3  # Up to 4 total attempts
    
    while expansion_iteration <= max_expansions:
        # Run grid search on current range
        optimal_L, optimal_value, all_values, optimal_idx = grid_search_1D(...)
        
        # Check if boundary hit
        if not hit_boundary:
            break  # Found optimal value within range
        
        # Expand range for next attempt
        expansion_factor = 3.0 ** (expansion_iteration + 1)
        if hit_lower_boundary:
            new_min = current_range[0] / expansion_factor
        else:  # hit_upper_boundary
            new_max = current_range[-1] * expansion_factor
        
        expansion_iteration += 1
```

### Step Size Grid Search
```python
def step_size_grid_search_with_expansion(...):
    # Set up step size range
    step_size_range = jnp.linspace(grid_min, grid_max, grid_size)
    
    # Boundary expansion loop (up to 4 attempts)
    expansion_iteration = 0
    max_expansions = 3  # Up to 4 total attempts
    
    while expansion_iteration <= max_expansions:
        # Run grid search on current range
        optimal_step_size, optimal_value, ... = grid_search_1D(...)
        
        # Check if boundary hit
        if not hit_boundary:
            break  # Found optimal value within range
        
        # Expand range for next attempt
        expansion_factor = 3.0 ** (expansion_iteration + 1)
        # ... expand range ...
        
        expansion_iteration += 1
```

## Example Output

```
=== Grid Search for L Parameter (with unadjusted step_size optimization) ===
Grid size: 10, Grid iterations: 2

--- Grid Iteration 1/2 (Initial Search) ---
  L range: [0.5000, 2.0000]
  Starting L grid search (outer loop)...
    L grid search attempt 1/4
      L 1/10: 0.5000 -> 45.2341
      L 2/10: 0.6667 -> 42.1234
      ...
      L 10/10: 2.0000 -> 48.9876
    Found optimal L 0.6667 within range
  L grid search complete!

--- Grid Iteration 2/2 (Refinement) ---
  Refining around previous optimal L: 0.6667
  L range: [0.5000, 0.8333]
  Starting L grid search (outer loop)...
    L grid search attempt 1/4
      L 1/10: 0.5000 -> 42.1234
      ...
      L 10/10: 0.8333 -> 43.5678
    Found optimal L 0.6667 within range
  L grid search complete!
```

## Boundary Expansion Example

```
--- Grid Iteration 1/2 (Initial Search) ---
  L range: [0.5000, 2.0000]
  Starting L grid search (outer loop)...
    L grid search attempt 1/4
      L 1/10: 0.5000 -> 45.2341  # Optimal hit lower boundary
      ...
    WARNING: Optimal L 0.5000 hit LOWER boundary 0.5000
    Expanding L grid range downward by factor 3...
    L grid search attempt 2/4
      L 1/10: 0.1667 -> 43.1234  # Still hitting boundary
      ...
    WARNING: Optimal L 0.1667 hit LOWER boundary 0.1667
    Expanding L grid range downward by factor 9...
    L grid search attempt 3/4
      L 1/10: 0.0556 -> 41.9876  # Found optimal within range
      ...
    Found optimal L 0.1667 within range
  L grid search complete!
```

## Configuration

### Default Parameters
- `grid_iterations=2`: Number of grid iterations
- `grid_size=6`: Number of points in each grid
- `max_expansions=3`: Maximum boundary expansions (4 total attempts)

### Customization
```python
# Use more iterations
grid_search_L_new(
    grid_iterations=3,  # 3 iterations instead of 2
    grid_size=15,       # 15 points instead of 10
    ...
)

# Use fewer iterations for speed
grid_search_L_new(
    grid_iterations=1,  # Single iteration
    grid_size=6,        # Smaller grid
    ...
)
```

## Error Handling

### Boundary Expansion Limits
If a boundary is still hit after 4 attempts:
```
ValueError: L grid search failed: still hitting lower boundary after 3 expansions. 
Consider manually adjusting the L range or model parameters.
```

### Infinite Values
If the optimal value is `inf` when hitting a boundary:
```
Optimal value is inf - returning boundary value without expansion
```

## Memory Management
- Cache is cleared every 5 evaluations in 1D grid search
- Cache is cleared every 3 L evaluations in main grid search
- Cache is cleared after each grid iteration
- Cache is cleared after each boundary expansion attempt

This ensures the grid search can handle large parameter spaces without memory issues. 