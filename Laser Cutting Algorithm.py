import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import matplotlib.patches as patches
import math

# Canvas and cutting area sizes
CANVAS_SIZE_X = 50
CANVAS_SIZE_Y = 50
CUTTING_AREA_X = 1000
CUTTING_AREA_Y = 200

global_margin = 0.5  # Optional global margin to space drawn shapes

# Initialize the main Tkinter window
root = tk.Tk()
root.title("Laser Cutting Shape Designer")

# Initialize a list to store points
points = []

# Function to handle clicks on the canvas
def on_canvas_click(event):
    x, y = event.x, event.y
    if 0 <= x < CANVAS_SIZE_X and 0 <= y < CANVAS_SIZE_Y:
        points.append((x, y))
        canvas.create_oval(x-2, y-2, x+2, y+2, fill="blue")
    else:
        messagebox.showerror("Error", "Point is out of bounds!")

# Function to calculate and display the convex hull
def calculate_convex_hull():
    if len(points) < 3:
        messagebox.showerror("Error", "At least 3 points are required to form a shape.")
        return
    
    points_array = np.array(points)
    hull = ConvexHull(points_array)

    # Generate placements with scaled points
    shape_points = points_array[hull.vertices]  # Points on the convex hull

# Function to calculate and display the convex hull
def calculate_convex_hull():
    if len(points) < 3:
        messagebox.showerror("Error", "At least 3 points are required to form a shape.")
        return
    
    points_array = np.array(points)
    hull = ConvexHull(points_array)

    # Generate placements with scaled points
    shape_points = points_array[hull.vertices]  # Points on the convex hull

    # Display convex hull plot first
    display_convex_hull_plot(points_array, hull)

    # Schedule the optimization and display of the cutting area to occur after the convex hull plot
    root.after(20, lambda: run_optimization(shape_points))

# Function to optimize placements and display the cutting area
def run_optimization(shape_points):
    placements, fabric_used, best_points = optimize_placement(shape_points)
    display_cutting_area(best_points, placements, fabric_used)
    usage_label.config(text=f"Fabric used: {fabric_used:.2f}%")

# Function to rotate the shape points by a given angle
def rotate_shape(shape_points, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
    return shape_points @ rotation_matrix.T

# Function to generate placements on the cutting area with horizontal optimization
def optimize_placement(shape_points):
    placements = []

    best_placements = []
    best_fabric_usage = 0
    
    shape_width, shape_height = calculate_polygon_dimensions(shape_points)
    ratio = math.ceil(shape_height/shape_width)
    
    #Checks if ratio is too narrow and throw error if so
    if ratio > 25:
        messagebox.showerror("Error", "Shape is too narrow for efficient placement on the cutting area.")
        return [], 0, []
    
    precision = math.ceil(interpolate(ratio, 1, 25, 3, 10))
    
    for i in range(360):
        rotated_points = rotate_shape(shape_points, angle=np.radians(i))
        
        best_x = find_best_fit_distance(rotated_points, direction="horizontal")
        best_y = find_best_fit_distance_scalable(rotated_points, best_x, precision, direction="vertical")
        
        current_x, current_y = -np.min(rotated_points[:, 0]) + global_margin, -np.min(rotated_points[:, 1]) + global_margin
        temp_placements = []

        while True:
            if current_y + np.max(rotated_points[:, 1]) >= CUTTING_AREA_Y:
                break
            
            # Record placement
            temp_placements.append((current_x, current_y))
            
            current_x += best_x  # Move to the next column
            
            if current_x + np.max(rotated_points[:, 0]) >= CUTTING_AREA_X:
                current_x = -np.min(rotated_points[:, 0]) + global_margin
                current_y += best_y  # Move down to the next row
        
        # Calculate fabric usage
        fabric_used = calculate_fabric_usage(rotated_points, len(temp_placements))

        # Update best placements if current usage is better
        if fabric_used > best_fabric_usage:
            best_fabric_usage = fabric_used
            best_placements = temp_placements
            best_shape_points = rotated_points  # Keep the best orientation for drawing

    return best_placements, best_fabric_usage, best_shape_points

def find_best_fit_distance(shape_points, direction="horizontal"):
    shifted_shape = np.copy(shape_points)  # Copy the original shape to start shifting
    shift_amount = 0  # Initialize the shift counter
    
    # Determine which coordinate to shift based on direction
    axis = 0 if direction == "horizontal" else 1

    # Start shifting and check for overlap
    while True:
        shift_amount += 1
        shifted_shape[:, axis] = shape_points[:, axis] + shift_amount - global_margin
        if not check_convex_overlap(shape_points, shifted_shape):
            break

    return shift_amount

def find_best_fit_distance_scalable(shape_points, shape_offset, precision, direction="vertical"):
    # Initialize the list of shapes by adding offsets to the initial shape_points
    shapes = [shape_points + np.array([shape_offset * i, 0]) for i in range(precision)]
    
    # Copy the initial shape to start shifting
    shifted_shapes = [np.copy(shape) for shape in shapes]
    
    shift_amount = 0  # Initialize the shift counter
    
    # Determine which coordinate to shift based on direction
    axis = 1 if direction == "vertical" else 0

    # Start shifting the copied shapes
    while True:
        shift_amount += 1
        # Apply shift to each copied shape
        for i in range(precision):
            shifted_shapes[i][:, axis] = shapes[i][:, axis] + shift_amount - interpolate(precision, 3, 10, global_margin/2, global_margin*2)
        
        # Check for overlap
        overlap_detected = False
        for i in range(precision):
            overlap_detected = check_convex_overlap(shapes[i], shifted_shapes[0]) or check_convex_overlap(shapes[0], shifted_shapes[i])
            if overlap_detected:
                break
        
        # If no overlap detected, exit loop
        if not overlap_detected:
            break

    return shift_amount

def interpolate(x, min_x, max_x, min_y, max_y):
    # Perform the interpolation
    y = min_y + (x - min_x) * (max_y - min_y) / (max_x - min_x)
    # Clamp the result to the [min_y, max_y] range
    return max(min_y, min(max_y, y))

def calculate_polygon_dimensions(polygon_points):
    # Ensure the points are in a numpy array
    polygon_points = np.array(polygon_points)
    
    # Center the polygon points by subtracting the mean (to reduce numerical errors)
    centroid = np.mean(polygon_points, axis=0)
    centered_points = polygon_points - centroid
    
    # Perform PCA (Principal Component Analysis)
    # Covariance matrix of the centered points
    cov_matrix = np.cov(centered_points.T)
    
    # Eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # The eigenvectors correspond to the principal axes of the polygon
    # The eigenvalues represent the variance (spread) along those axes.
    
    # Project the points onto the principal axes (eigenvectors)
    projections = np.dot(centered_points, eigenvectors)
    
    # The range (min, max) along each principal axis gives the dimensions
    width = projections[:, 0].max() - projections[:, 0].min()
    height = projections[:, 1].max() - projections[:, 1].min()
    
    return width, height

def check_convex_overlap(shape1_points, shape2_points):
    # Create polygons from the point sets
    polygon1 = Polygon(shape1_points)
    polygon2 = Polygon(shape2_points)

    # This checks for any intersection
    return polygon1.intersects(polygon2)

# Function to calculate fabric usage percentage
def calculate_fabric_usage(shape_points, num_placements):
    # Calculate the area of the convex hull
    shape_area = 0.5 * np.abs(np.dot(shape_points[:, 0], np.roll(shape_points[:, 1], 1)) -
                               np.dot(shape_points[:, 1], np.roll(shape_points[:, 0], 1)))
    total_shape_area = shape_area * num_placements
    cutting_area_total = CUTTING_AREA_X * CUTTING_AREA_Y
    fabric_used_percentage = (total_shape_area / cutting_area_total) * 100
    return fabric_used_percentage

# Function to display the cutting area layout
def display_cutting_area(best_points, placements, fabric_used):
    plt.figure(figsize=(10, 2))
    plt.xlim(0, CUTTING_AREA_X)
    plt.ylim(0, CUTTING_AREA_Y)
    
    for (x, y) in placements:
        hull = ConvexHull(best_points)
        shape_points = best_points[hull.vertices]  # Points on the convex hull
        # Translate shape points by the (x, y) placement
        translated_shape = shape_points + np.array([x, y])
        
        # Create a Polygon object and add it to the plot
        polygon = patches.Polygon(translated_shape, closed=True, color='blue', alpha=0.5)
        plt.gca().add_patch(polygon)

    plt.title("Optimized Cutting Layout")
    plt.gca().invert_yaxis()
    
    # Add fabric used text to the plot
    plt.text(10, CUTTING_AREA_Y - CANVAS_SIZE_Y, f"Fabric used: {fabric_used:.2f}%", fontsize=12, color='black')
    
    plt.show(block = False)

# Function to display convex hull plot
def display_convex_hull_plot(points_array, hull):
    # Descale the points to show original shape size
    points_array = points_array / (200 / 5) 
    
    plt.figure()
    plt.plot(points_array[:, 0], points_array[:, 1], 'o')
    
    # Draw the convex hull
    for simplex in hull.simplices:
        plt.plot(points_array[simplex, 0], points_array[simplex, 1], 'k-')
    plt.title("Convex Hull of the Shape")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.gca().invert_yaxis()
    plt.show(block = False)

# Canvas for drawing
canvas = tk.Canvas(root, width=CANVAS_SIZE_X, height=CANVAS_SIZE_Y, bg="white")
canvas.bind("<Button-1>", on_canvas_click)
canvas.pack()

# Button to calculate convex hull
calculate_button = tk.Button(root, text="Calculate Convex Hull", command=calculate_convex_hull)
calculate_button.pack()

# Label to display fabric usage
usage_label = tk.Label(root, text="Fabric used: 0%")
usage_label.pack()

root.mainloop()