const std = @import("std");
const allocator = std.mem.allocator;
const List = std.ArrayList([][]f32);
pub const Vector = struct {
    data: []f32,

    pub fn init(data: []f32) Vector {
        return Vector{
            .data = data,
        };
    }

};

pub const Minor = struct {
    minor: f32,
    tensor: [][]f32,

    pub fn init(minor: f32, tensor: anytype) Minor {
        return Minor{
            .minor = minor,
            .tensor = tensor,
        };
    }

    pub fn getMinor(self: *const Minor) f32 {
        return self.minor;
    }

    pub fn getTensor(self: *const Minor) [][]f32 {
        return self.tensor;
    }

    pub fn setMinor(self: *Minor, new_minor: f32) void {
        self.minor = new_minor;
    }
};

pub const Matrix = struct {
    tensor: [][]f32,
    vectors: std.ArrayList(Vector),
    allocator: std.mem.Allocator,
    bias:f32,
    slopes:[]f32,

    pub fn init(allocator: std.mem.Allocator, tensor: [][]f32) Matrix {
        return Matrix{
            .tensor = tensor,
            .vectors = std.ArrayList(Vector).init(allocator),
            .allocator = allocator,
            .bias = 0.0,
            .slopes = undefined,
        };
    }

    pub fn deinit(self: *Matrix) void {
        self.vectors.deinit();
    }

    // Getter
    pub fn getTensor(self: *const Matrix) [][]f32 {
        return self.tensor;
    }

    // Setter
    pub fn setTensor(self: *Matrix, t: anytype) void {
        self.tensor = t;
    }

    // Ajouter un vecteur
    pub fn add(self: *Matrix, allocator: std.mem.Allocator, vector: Vector) !void {
        const new_len = self.tensor.len + 1;
        var new_tensor = try allocator.realloc([]f32, self.tensor, new_len);
        new_tensor[new_len - 1] = vector;
        self.tensor = new_tensor;
        try self.vectors.append(vector);
    }

    // Ajouter avec biais
    pub fn addVector(self: *Matrix, v: Vector) !void {
        try self.add(allocator,v);
        self.generateBias(self.vectors);
    }

    // Calcul du déterminant
    fn detMatrix(self: *Matrix, mat: anytype) !f32 {
        const len = mat.len;
        if (len == 0) return 1;
        if (len == 1) return mat[0][0];
        if (len == 2) return self.det2x2(mat);

        var sum: f32 = 0.0;
        var sign: f32 = 1.0;

        for (mat[0], 0..) |val, i| {
            var sub_matrix = try self.allocator.alloc([]f32, len - 1);
            defer self.allocator.free(sub_matrix);

            for (0..len - 1) |row_index| {
                sub_matrix[row_index] = try self.allocator.alloc(f32, len - 1);
                var col_count: usize = 0;
                for (0..len) |col_index| {
                    if (col_index == i) continue;
                    sub_matrix[row_index][col_count] = mat[row_index + 1][col_index];
                    col_count += 1;
                }
            }
            sum += sign * val * try self.detMatrix(sub_matrix);
            sign *= -1;
        }

        return sum;
    }

    // Déterminant 2x2
    fn det2x2(mat:anytype) f32 {
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    }

    // from a matrix delete the first element from each array
    fn pop(allocator: std.mem.Allocator, arr: []f32) ![]f32 {
        if (arr.len <= 1) return error.EmptyArray;
        var popped = try allocator.alloc(f32, arr.len - 1);
        for (popped, arr[1..]) |*item, val| {
            item.* = val;
        }
        return popped;
    }
    //retrieve minors from a matrix which correspond to the first using of pop function
    fn minorCorrespondingTensor(
        minor: f32,
        t: anytype,
        allocator: std.mem.Allocator,
    ) !@TypeOf(t) {
        var mat = std.ArrayList([]f32).init(allocator);
        defer mat.deinit();

        for (t) |row| {
            if (row[0] != minor) {
                const new_row = try pop(allocator, row);
                try mat.append(new_row);
            }
        }

        return mat.toOwnedSlice();
    }

    fn minorsFromTensor(
        tensor: [][]f32,
        allocator: std.mem.Allocator,
    ) ![]Minor {
        const len = tensor.len;
        var minors = try allocator.alloc(Minor, len);

        for (tensor, 0..len) |row, i| {
            const m_tensor = try minorCorrespondingTensor(row[0], tensor, allocator);
            minors[i] = Minor.init(row[0], m_tensor);
        }

        return minors;
    }

    fn setMinorValue(minors: []Minor) void {
        var sign: bool = false; // false = positif, true = négatif
        for (minors) |*m| {
            if (sign) {
                m.minor = -m.minor;
            }
            sign = !sign;
        }
    }

    pub fn trace(self: *Matrix) f32 {
        var sum: f32 = 0.0;
        var i: usize = 0;
        const len = self.tensor.len;
        for (self.vectors.items) |vector| {
            sum = sum + vector.data[i];
            i += 1;
        }
        return sum;
    }

    pub fn vars(self: *Matrix, b: anytype) !@TypeOf(b) {
        const det_A = try self.detMatrix(self.tensor);
        if (det_A == 0) {
            return error.SingularMatrix;
        }

        var result = try self.allocator.alloc(f32, self.tensor.len);
        errdefer self.allocator.free(result);

        for (0..self.tensor.len) |i| {
            var temp_matrix = try self.cloneMatrix();
            defer self.freeMatrix(temp_matrix);

            for (0..self.tensor.len) |j| {
                temp_matrix[j][i] = b[j];
            }

            result[i] = (try self.detMatrix(temp_matrix)) / det_A;
        }

        return result;
    }

    fn cloneMatrix(self: *Matrix) ![][]f32 {
        var new_matrix = try self.allocator.alloc([]f32, self.tensor.len);
        errdefer {
            for (new_matrix) |row| self.allocator.free(row);
            self.allocator.free(new_matrix);
        }

        for (self.tensor, 0..) |row, i| {
            new_matrix[i] = try self.allocator.dupe(f32, row);
        }
        return new_matrix;
    }

    fn freeMatrix(self: *Matrix, matrix: anytype) void {
        for (matrix) |row| self.allocator.free(row);
        self.allocator.free(matrix);
    }

    // Autres méthodes : pop, trace, generateBias, minorsFromTensor, etc.
    fn generateBias(vec: anytype) f32 {
        var sum: f32 = 0.0;
        for (vec) |val| {
            sum += val * val;
        }
        return std.math.sqrt(sum);
    }
    
    fn generate(self:*Matrix,allocator:std.mem.Allocator, mat:std.ArrayList(Vector)) !void {
        var bias = 0.0;
        var list = List.init(allocator);
        for(mat.items) |v| {
            try list.append(v.data);
        }
        ml.generateBiasAndSlopes(list,self.bias,self.slopes);
    }
};

pub const FlatMatrix = struct {
    data: [][]f32,
    rows: usize,
    cols: usize,
    
    pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !FlatMatrix {
        return FlatMatrix{
            .data = try allocator.alloc(f32, rows * cols),
            .rows = rows,
            .cols = cols,
        };
    }
    
    pub fn deinit(self: *FlatMatrix, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }
    
    // Accès efficace avec calcul d'index
    pub fn get(self: FlatMatrix, row: usize, col: usize) f32 {
        return self.data[row * self.cols + col];
    }
    
    pub fn set(self: *FlatMatrix, row: usize, col: usize, value: f32) void {
        self.data[row * self.cols + col] = value;
    }
    
    // Opérations matricielles optimisées
    pub fn transpose(self: FlatMatrix, allocator: std.mem.Allocator) !FlatMatrix {
        var transposed = try FlatMatrix.init(allocator, self.cols, self.rows);
        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                transposed.set(j, i, self.get(i, j));
            }
        }
        return transposed;
    }
};

pub const Point = struct {
    x: f32,
    y: f32,
    z: f32,
};

pub const Dataset = struct {
    point: Point,
    xs: []f32,
    ys: []f32,
    zs: []f32,
    alpha_s: []f32,
    sw: f32,
    dim: u8,
};

pub fn weights(allocator: std.mem.Allocator, inputs: [][]f32, d: Dataset, it: i32) ![][]f32 {
    var weights_slice = try allocator.alloc([]f32, inputs.len);
    errdefer {
        for (weights_slice) |row| allocator.free(row);
        allocator.free(weights_slice);
    }

    for (inputs, 0..) |input_row, i| {
        var result = try allocator.alloc(f32, input_row.len);
        // NOTE: No defer here, since we are transferring ownership to weights_slice

        for (input_row, 0..) |_, j| {
            const w = d.ys[j] - predicted(d.sw, input_row[j], d, it);
            result[j] = w;
        }
        weights_slice[i] = result;
    }
    return weights_slice;
}

pub fn checkDependency(self: *LinearDependency, tensor: [][]f32) void {
        var check: i8 = 0;
        for (tensor, 0..) |v1, i| {
            for (tensor[i + 1 ..]) |v2| {
                if (self.compare(v1, v2)) {
                    check += 1;
                }
            }
        }
        self.satisfiedCase = check;
    }
    ///compare to array if they are lineary independent <=> v1 = av2 where a is i32 or reductible float
    pub fn compare(self: *LinearDependency, v1: []const f32, v2: []const f32) bool {
        if (v1.len != v2.len or v1.len == 0) return false;

        var factor: ?f32 = null;
        for (v1, v2) |a, b| {
            if (a == 0 and b == 0) continue;
            if (a == 0 or b == 0) return false; // One is zero, the other is not

            const current_factor = a / b;
            if (factor) |f| {
                if (f != current_factor) return false;
            } else {
                factor = current_factor;
            }
        }
        return true;
    }
    
fn closest_point(points: []const Point, centroids: []const Point, point_index: usize) usize {
    var min_dist: f32 = -1;
    var closest_centroid_index: usize = 0;

    for (centroids, 0..) |centroid, i| {
        const dist = euclideanDistance(points[point_index], centroid);
        if (min_dist == -1 or dist < min_dist) {
            min_dist = dist;
            closest_centroid_index = i;
        }
    }

    return closest_centroid_index;
}

fn nearest_neighboors(d: Dataset, p: Point, nearest_neighbors:List(f32), min_dist:f32) usize {
    var nearests_neighbor_index: usize = 0;
    for (d.points, 0..) |neighbor, i| {
         const dist = euclideanDistance(p, neighbor);
         if (dist <= min_dist) {
             min_dist = dist;
             nearest_neighbors_index = i;
             nearest_neighbors.items[nearest_neighbors.items.len] = nearest_neighbors_index;
         }
    }

    return nearest_neighbors.items.len;
}

fn euclideanDistance(p1: Point, p2: Point) f32 {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    const dz = p1.z - p2.z;
    return @sqrt(dx * dx + dy * dy + dz * dz);
}


fn dist(datasets: []const Dataset, points: []const Point, centroids: []const Point, labels: []i32) void {
    for (datasets, 0..) |_, i| {
        labels[i] = @intCast(i32, closest_point(points, centroids, i));
    }
}

const Strings = struct {
    allocator: Allocator,
    str: []u8,

    pub fn new(allocator: Allocator, value: []const u8) !String {
        const str = try allocator.dupe(u8, value);
        return String{ .allocator = allocator, .str = str };
    }

    pub fn deinit(self: *String) void {
        self.allocator.free(self.str);
    }

    pub fn char_at(self: *String, index: usize) u8 {
        return self.str[index];
    }

    pub fn index_of(self: *String, c: u8) u8 {
        return std.mem.indexOf(u8, self.str, c) != -1;
    }

    pub fn equals(self: *String, value: []const u8) bool {
        return std.mem.eql(u8, self.str, @constCast(value));
    }

    pub fn contains(self: String, value: []const u8) bool {
        return mem.contains(u8, self.str, value);
    }

    pub fn concat(self: String, seq:[]const u8) ![]u8 {
	      var buffer: [_]u8 = undefined;
        const newSeq = try std.mem.concat(u8, &buffer, &[[]const u8]{self.str, seq});
	      return newSeq;
    }

    pub fn substr(self: String, start: usize, end: usize) ![]u8 {
        if (end > self.str.len) {
            return error.OutOfBounds;
        }
        return self.str[start..end];
    }
    
    pub fn replace(self: *String, old: []const u8, new: []const u8) !void {
        const new_str = try std.mem.replaceOwned(u8, self.allocator, self.str, old, new);
        self.allocator.free(self.str);
        self.str = new_str;
    }


};