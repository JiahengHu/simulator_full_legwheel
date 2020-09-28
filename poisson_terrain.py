import itertools
import math
import numpy as np
# import pybullet as p
import os
# import pybullet_data

_GRID_LENGTH = 10
_GRID_WIDTH = 4
_MAX_SAMPLE_SIZE = 10
_MIN_BLOCK_HEIGHT = 0
_MAX_BLOCK_LENGTH = 1
_MIN_BLOCK_LENGTH = 0.1

# # high and large and dense
# _MIN_BLOCK_DISTANCE_LOW = 0.3
# _MAX_BLOCK_HEIGHT_HIGH = 0.15

# # shallow and small and sparse
# _MIN_BLOCK_DISTANCE_HIGH = 1
# _MAX_BLOCK_HEIGHT_LOW = 0.05


class PoissonDisc2D(object):
  """Generates 2D points using Poisson disk sampling method.

  Implements the algorithm described in:
    http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
  Unlike the uniform sampling method that creates small clusters of points,
  Poisson disk method enforces the minimum distance between points and is more
  suitable for generating a spatial distribution of non-overlapping objects.
  """

  def __init__(self, grid_length, grid_width, min_radius, max_sample_size):
    """Initializes the algorithm.

    Args:
      grid_length: The length of the bounding square in which points are
        sampled.
      grid_width: The width of the bounding square in which points are
        sampled.
      min_radius: The minimum distance between any pair of points.
      max_sample_size: The maximum number of sample points around a active site.
        See details in the algorithm description.
    """
    self._cell_length = min_radius / math.sqrt(2)
    self._grid_length = grid_length
    self._grid_width = grid_width
    self._grid_size_x = int(grid_length / self._cell_length) + 1
    self._grid_size_y = int(grid_width / self._cell_length) + 1
    self._min_radius = min_radius
    self._max_sample_size = max_sample_size
    self.ground_id = 0

    # Flattern the 2D grid as an 1D array. The grid is used for fast nearest
    # point searching.
    self._grid = [None] * self._grid_size_x * self._grid_size_y

    # Generate the first sample point and set it as an active site.
    first_sample = np.array(
        np.random.random_sample(2)) * [grid_length, grid_width]
    self._active_list = [first_sample]

    # Also store the sample point in the grid.
    self._grid[self._point_to_index_1d(first_sample)] = first_sample


  def _point_to_index_1d(self, point):
    """Computes the index of a point in the grid array.

    Args:
      point: A 2D point described by its coordinates (x, y).

    Returns:
      The index of the point within the self._grid array.
    """
    return self._index_2d_to_1d(self._point_to_index_2d(point))

  def _point_to_index_2d(self, point):
    """Computes the 2D index (aka cell ID) of a point in the grid.

    Args:
      point: A 2D point (list) described by its coordinates (x, y).

    Returns:
      x_index: The x index of the cell the point belongs to.
      y_index: The y index of the cell the point belongs to.
    """
    x_index = int(point[0] / self._cell_length)
    y_index = int(point[1] / self._cell_length)
    return x_index, y_index

  def _index_2d_to_1d(self, index2d):
    """Converts the 2D index to the 1D position in the grid array.

    Args:
      index2d: The 2D index of a point (aka the cell ID) in the grid.

    Returns:
      The 1D position of the cell within the self._grid array.
    """
    return index2d[0] + index2d[1] * self._grid_size_x

  def _is_in_grid(self, point):
    """Checks if the point is inside the grid boundary.

    Args:
      point: A 2D point (list) described by its coordinates (x, y).

    Returns:
      Whether the point is inside the grid.
    """
    return (0 <= point[0] < self._grid_length) and (0 <= point[1] <
                                                    self._grid_width)

  def _is_in_range(self, index2d):
    """Checks if the cell ID is within the grid.

    Args:
      index2d: The 2D index of a point (aka the cell ID) in the grid.

    Returns:
      Whether the cell (2D index) is inside the grid.
    """

    return (0 <= index2d[0] < self._grid_size_x) and (0 <= index2d[1] <
                                                      self._grid_size_y)

  def _is_close_to_existing_points(self, point):
    """Checks if the point is close to any already sampled (and stored) points.

    Args:
      point: A 2D point (list) described by its coordinates (x, y).

    Returns:
      True iff the distance of the point to any existing points is smaller than
      the min_radius
    """
    px, py = self._point_to_index_2d(point)
    # Now we can check nearby cells for existing points
    for neighbor_cell in itertools.product(
        range(px - 1, px + 2), range(py - 1, py + 2)):

      if not self._is_in_range(neighbor_cell):
        continue

      maybe_a_point = self._grid[self._index_2d_to_1d(neighbor_cell)]
      if maybe_a_point is not None and np.linalg.norm(
          maybe_a_point - point) < self._min_radius:
        return True

    return False

  def sample(self):
    """Samples new points around some existing point.

    Removes the sampling base point and also stores the new jksampled points if
    they are far enough from all existing points.
    """
    active_point = self._active_list.pop()
    for _ in range(self._max_sample_size):
      # Generate random points near the current active_point between the radius
      random_radius = np.random.uniform(self._min_radius, 2 * self._min_radius)
      random_angle = np.random.uniform(0, 2 * math.pi)

      # The sampled 2D points near the active point
      sample = random_radius * np.array(
          [np.cos(random_angle), np.sin(random_angle)]) + active_point

      if not self._is_in_grid(sample):
        continue

      if self._is_close_to_existing_points(sample):
        continue

      self._active_list.append(sample)
      self._grid[self._point_to_index_1d(sample)] = sample

  def generate(self):
    """Generates the Poisson disc distribution of 2D points.

    Although the while loop looks scary, the algorithm is in fact O(N), where N
    is the number of cells within the grid. When we sample around a base point
    (in some base cell), new points will not be pushed into the base cell
    because of the minimum distance constraint. Once the current base point is
    removed, all future searches cannot start from within the same base cell.

    Returns:
      All sampled points. The points are inside the quare [0, grid_length] x [0,
      grid_width]
    """

    while self._active_list:
      self.sample()

    all_sites = []
    for p in self._grid:
      if p is not None:
        all_sites.append(p)

    return all_sites


class TerrainRandomizer():
  """Generates an uneven terrain in the bullet."""
  def __init__(
      self):
    self.block_IDs = []
    
    """Initializes the randomizer.

    """
  def reset(self):#, pybullet_client):

    self.block_IDs = []

    # while len( self.block_IDs)>0:
    #     block_ID = self.block_IDs.pop()
    #     pybullet_client.removeBody(block_ID)

  def randomize_env(self, pybullet_clients, _MIN_BLOCK_DISTANCE, _MAX_BLOCK_HEIGHT):
    """Generate a random terrain for the current env.

    """
    self._MAX_BLOCK_HEIGHT_FOR_COLOR = 0.1

    self._MIN_BLOCK_DISTANCE = _MIN_BLOCK_DISTANCE
    self._MAX_BLOCK_HEIGHT = _MAX_BLOCK_HEIGHT
    self.block_IDs , self.block_centers, self.half_length1_list,\
      self.half_length2_list, self.half_height_list = self._generate_convex_blocks(pybullet_clients)

  def _generate_convex_blocks(self,pybullet_clients):
    """Adds random convex blocks to the flat ground.

    We use the Possion disk algorithm to add some random blocks on the ground.
    Possion disk algorithm sets the minimum distance between two sampling
    points, thus voiding the clustering effect in uniform N-D distribution.


    """

    block_IDs = [list() for i in range(len(pybullet_clients))]

    poisson_disc = PoissonDisc2D(_GRID_LENGTH, _GRID_WIDTH, self._MIN_BLOCK_DISTANCE,
                                 _MAX_SAMPLE_SIZE)

    block_centers_out = poisson_disc.generate()
    half_length1_list = list()
    half_length2_list = list()
    half_height_list = list()
    block_centers = list()

    for center in block_centers_out:
      # We want the blocks to be in front of the robot.
      shifted_center = np.array(center) - [1, _GRID_WIDTH / 2]

      # Do not place blocks near the point [0, 0], where the robot will start.
      # if abs(shifted_center[0]) < 0.5 and abs(shifted_center[1]) < 0.5:
      if np.linalg.norm(shifted_center[0:1]) < 0.5:
        continue
      else:
        block_centers.append(center)

      half_length1 = np.random.uniform(_MIN_BLOCK_LENGTH, _MAX_BLOCK_LENGTH) / (
          2 * math.sqrt(2))
      half_length2 = np.random.uniform(_MIN_BLOCK_LENGTH, _MAX_BLOCK_LENGTH) / (
          2 * math.sqrt(2))
      half_height = np.random.uniform(_MIN_BLOCK_HEIGHT, self._MAX_BLOCK_HEIGHT) / 2
      
      half_length1_list.append(half_length1)
      half_length2_list.append(half_length2)
      half_height_list.append(half_height)


      # add the block to all the envs
      for i_env in range(len(pybullet_clients)):
        pybullet_client = pybullet_clients[i_env]

        box_id = pybullet_client.createCollisionShape(
            pybullet_client.GEOM_BOX,
            halfExtents=[half_length1, half_length2, half_height])
        block_color = 1 - np.interp(2*half_height,
               [ _MIN_BLOCK_HEIGHT, self._MAX_BLOCK_HEIGHT_FOR_COLOR], [0,1]) # scales 0-1 for block color
        visual_id = pybullet_client.createVisualShape(
            pybullet_client.GEOM_BOX,
            rgbaColor = [block_color, block_color, block_color, 1],
            halfExtents=[half_length1, half_length2, half_height])
        block_ID = pybullet_client.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box_id,
            baseVisualShapeIndex = visual_id,
            basePosition=[shifted_center[0], shifted_center[1], half_height])
        block_IDs[i_env].append(block_ID) 

    # print(block_centers)
    # print(half_height_list)
    return block_IDs, block_centers, half_length1_list,half_length2_list,half_height_list

              # rgbaColor = [0.7*block_color, 0.7*block_color, 1.0*block_color, 1],

  def alter_block_heights(self,pybullet_clients, delta_height):
    # for i_env in range(len(pybullet_clients)):
    #     block_IDs_i = self.block_IDs[i_env]
    #     pybullet_client = pybullet_clients[i_env]
    #     for block_ID in block_IDs_i:
    #       pybullet_client.removeBody(block_ID)
    block_IDs = [list() for i in range(len(pybullet_clients))]

    for ib in range(len(self.block_centers)):
      center = self.block_centers[ib]
      shifted_center = np.array(center) - [1, _GRID_WIDTH / 2]

      half_length1 = self.half_length1_list[ib]
      half_length2 = self.half_length2_list[ib]
      self.half_height_list[ib] = max(0, self.half_height_list[ib] + delta_height)
      half_height = self.half_height_list[ib]

      # add the block to all the envs
      for i_env in range(len(pybullet_clients)):
        pybullet_client = pybullet_clients[i_env]

        box_id = pybullet_client.createCollisionShape(
            pybullet_client.GEOM_BOX,
            halfExtents=[half_length1, half_length2, half_height])
        block_color = 1 - np.interp(2*half_height,
               [ _MIN_BLOCK_HEIGHT, self._MAX_BLOCK_HEIGHT_FOR_COLOR], [0,1]) # scales 0-1 for block color
        visual_id = pybullet_client.createVisualShape(
            pybullet_client.GEOM_BOX,
            rgbaColor = [block_color, block_color, block_color, 1],
            halfExtents=[half_length1, half_length2, half_height])
        block_ID = pybullet_client.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box_id,
            baseVisualShapeIndex = visual_id,
            basePosition=[shifted_center[0], shifted_center[1], half_height])
        block_IDs[i_env].append(block_ID) 
    self.block_IDs = block_IDs
    
