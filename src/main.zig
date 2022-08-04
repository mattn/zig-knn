const std = @import("std");

fn distance(lhs: [4]f64, rhs: [4]f64) f64 {
    var sum: f64 = 0;
    var i: usize = 0;
    while (i < lhs.len) {
        sum += std.math.pow(f64, lhs[i] - rhs[i], 2);
        i += 1;
    }
    return sum;
}

const item = struct {
    i: usize,
    f: f64,
};

const rank = struct {
    i: i64,
    s: []const u8,
};

const KNN = struct {
    a: std.mem.Allocator,
    k: usize,
    XX: [][4]f64,
    Y: [][]const u8,

    const Self = @This();

    fn init(a: std.mem.Allocator, k: usize, XX: [][4]f64, Y: [][]const u8) Self {
        return Self{
            .a = a,
            .k = k,
            .XX = XX,
            .Y = Y,
        };
    }

    fn predict(self: Self, X: [][4]f64, r: [][]const u8) !void {
        var i: usize = 0;
        while (i < X.len) {
            var x = X[i];
            const items = try self.a.alloc(item, self.XX.len);
            defer self.a.free(items);

            var j: usize = 0;
            while (j < self.XX.len) {
                var xx = self.XX[j];
                items[j] = .{
                    .i = i,
                    .f = distance(x, xx),
                };
                j += 1;
            }
            const ff = struct {
                fn f(comptime _: type, a: anytype, b: anytype) bool {
                    return a.f < b.f;
                }
            }.f;

            std.sort.sort(item, items[0..self.XX.len], void, ff);

            const labels = try self.a.alloc([]const u8, self.k);
            defer self.a.free(labels);

            j = 0;
            while (j < self.k) {
                labels[j] = self.Y[items[i].i];
                j += 1;
            }

            var found = std.StringArrayHashMap(i32).init(self.a);
            j = 0;
            while (j < self.k) {
                var v = found.get(labels[j]);
                if (v != null) {
                    try found.put(labels[j], v.? + 1);
                } else {
                    try found.put(labels[j], 0);
                }
                j += 1;
            }

            const ranks = try self.a.alloc(rank, found.keys().len);
            defer self.a.free(ranks);

            j = 0;
            for (found.keys()) |k| {
                var v = found.get(k);
                ranks[j] = .{
                    .i = v.?,
                    .s = k,
                };
                j += 1;
            }
            const fi = struct {
                fn f(comptime _: type, a: anytype, b: anytype) bool {
                    return a.i < b.i;
                }
            }.f;

            std.sort.sort(item, items[0..self.k], void, fi);
            r[i] = ranks[0].s;

            i += 1;
        }
    }
};

pub fn main() anyerror!void {
    var allocator = std.heap.page_allocator;

    var f = try std.fs.cwd().openFile("iris.csv", .{});
    defer f.close();

    var trainX = std.ArrayList([4]f64).init(allocator);
    defer trainX.deinit();
    var trainY = std.ArrayList([]const u8).init(allocator);
    defer trainY.deinit();

    var testX = std.ArrayList([4]f64).init(allocator);
    defer testX.deinit();
    var testY = std.ArrayList([]const u8).init(allocator);
    defer testY.deinit();

    var c: usize = 0;
    _ = try f.reader().readUntilDelimiterAlloc(allocator, '\n', 1024);
    while (f.reader().readUntilDelimiterAlloc(allocator, '\n', 1024)) |line| {
        var iter = std.mem.split(u8, line, ",");

        if (c % 2 == 0) {
            try trainX.append([4]f64{
                try std.fmt.parseFloat(f64, iter.next().?),
                try std.fmt.parseFloat(f64, iter.next().?),
                try std.fmt.parseFloat(f64, iter.next().?),
                try std.fmt.parseFloat(f64, iter.next().?),
            });
            try trainY.append(iter.next().?);
        } else {
            try testX.append([4]f64{
                try std.fmt.parseFloat(f64, iter.next().?),
                try std.fmt.parseFloat(f64, iter.next().?),
                try std.fmt.parseFloat(f64, iter.next().?),
                try std.fmt.parseFloat(f64, iter.next().?),
            });
            try testY.append(iter.next().?);
        }
        c += 1;
    } else |err| {
        if (err != error.EndOfStream) {
            std.log.warn("{}", .{err});
        }
    }

    var knn = KNN.init(allocator, 8, trainX.items, trainY.items);

    const results = try allocator.alloc([]const u8, testY.items.len);
    defer allocator.free(results);

    try knn.predict(
        testX.items,
        results,
    );

    var i: usize = 0;
    var hit: usize = 0;
    while (i < testY.items.len) {
        if (std.mem.eql(u8, testY.items[i], results[i])) {
            hit += 1;
        }
        i += 1;
    }
    std.log.warn("{d:.2}", .{@intToFloat(f64, hit) / @intToFloat(f64, trainY.items.len) * 100});
}
