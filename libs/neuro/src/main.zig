const std = @import("std");
const testing = std.testing;

const Op = enum(u8) {
    add,
    mul,
    sub,
    div,
    pow,
    // non-linearn
    relu,
    tanh,
    none
};

pub fn Value(comptime T: type) type {
    return struct {
        const Self = @This();

        data: T,
        grad: T = 0,
        // The operation that created this value
        op: Op = .none,
        _prev: [2]?*Self = .{null} ** 2,

        pub fn init(value: T) Self {
            var self = Self{
                .data = value, 
            }; 
            return self;
        }
        pub fn add(self: *Self, other: *Self) Self {
            return Self{
                .data = self.data + other.data,
                .op = .add,
                ._prev = .{self, other}
            }; 
        }

        pub fn sub(self: *Self, other: *Self) Self {
            return Self{
                .data = self.data - other.data,
                .op = .sub,
                ._prev = .{self, other}
            }; 
        }

        pub fn relu(self: *Self) Self {
            return Self{
                .data = if (self.data > 0) self.data else 0,
                .op = .relu,
                ._prev = .{self, null},
            };
        }
        pub fn tanh(self: *Self) Self {
            const exp2 = @exp(self.data * 2);
            return Self{
                .data = (exp2 - 1)/(exp2 + 1),
                .op = .tanh,
                ._prev = .{self, null},
            };
        }

        pub fn mul(self: *Self, other: *Self) Self {
            return Self{
                .data = self.data * other.data,
                .op = .mul,
                ._prev = .{self, other}
            }; 
        }

        fn buildTopo(
            self: *Self, 
            topo: *std.ArrayList(*Self),
            visited: *std.AutoArrayHashMap(*Self, void),
        ) std.mem.Allocator.Error!void {
            if (visited.get(self) == null) {
                try visited.put(self, .{});
                for (self._prev) |o_child| {
                    if (o_child) |child| {
                        try child.buildTopo(topo, visited);
                    }
                }
                try topo.append(self);
            }
        }

        pub fn toGraphViz(self: *Self, alloc: std.mem.Allocator) !std.ArrayList(u8) {
            var str_builder = std.ArrayList(u8).init(alloc);
            var writer = str_builder.writer();
            _ = try writer.write("\ndigraph nodes {\n\tnode [shape=record];\n");

            var visited = std.AutoArrayHashMap(*Self, void).init(alloc);
            defer visited.deinit();

            var topo = std.ArrayList(*Self).init(alloc);
            defer topo.deinit();

            try self.buildTopo(&topo, &visited);
            for (topo.items) |node| {
                const op_str = switch (node.op) {
                    .add => "+",
                    .sub => "-",
                    .mul => "*",
                    .div => "/",
                    .pow => "^",
                    .relu => "ReLU",
                    .tanh => "tanh",
                    .none => "=",
                };
                _ = try writer.print("\t\"{*}\" [label=\"data={d}|grad={d}\"];\n", 
                    .{node, node.data, node.grad});
                if (node.op != .none) {
                    _ = try writer.print("\t\"{*}-op\" [label=\"{s}\"];\n",
                        .{node, op_str});
                    _ = try writer.print("\t\"{0*}-op\" -> \"{0*}\";\n", .{node});
                }
                for (node._prev) |o_child| {
                    if (o_child) |child| {
                        _ = try writer.print("\t\"{*}\" -> \"{*}-op\"\n", .{child, node});
                    }
                }
            }
            _ = try writer.write("}\n");
            return str_builder;
        }

        pub fn backward(self: *Self, alloc: std.mem.Allocator) std.mem.Allocator.Error!void {
            var visited = std.AutoArrayHashMap(*Self, void).init(alloc);
            defer visited.deinit();

            var topo = std.ArrayList(*Self).init(alloc);
            defer topo.deinit();

            try self.buildTopo(&topo, &visited);

            self.grad = 1;
            var i : usize = topo.items.len;
            while (i > 0) : (i -= 1) {
                var node = topo.items[i-1];
                node.backwardHandler(node.op);
            }
        }

        fn backwardHandler(self: *const Self, op: Op) void {
            var left = self._prev[0];
            var right = self._prev[1];
            switch (op) {
                .add => {
                    left.?.grad += self.grad;            
                    right.?.grad += self.grad;
                },
                .sub => {
                    left.?.grad += self.grad;            
                    right.?.grad -= self.grad;
                },
                .mul => {
                    left.?.grad += right.?.data * self.grad;            
                    right.?.grad += left.?.data * self.grad;
                },
                .relu => {
                    left.?.grad += if (self.data > 0) self.grad else 0;
                },
                .tanh => {
                    left.?.grad += (1 - (self.data * self.data)) * self.grad;
                },
                .none => {},
                else => @panic("Unimplemented backward operation"),
            }
        }
    };
}

test "value init" {
    var val = Value(f16).init(10.0);
    try testing.expectEqual(@as(f16, 10.0), val.data);
    try testing.expectEqual(@as(f16, 0), val.grad);
}

test "value add" {
    var a = Value(f16).init(10.0);
    var b = Value(f16).init(12.0);
    // forward pass
    var c = a.add(&b);
    try testing.expectEqual(@as(f16, 22.0), c.data);
    // backward pass
    try c.backward(std.testing.allocator);
    try testing.expectEqual(@as(f16, 1.0), c.grad);
    try testing.expectEqual(@as(f16, 1.0), a.grad);
    try testing.expectEqual(@as(f16, 1.0), b.grad);
}

test "value sub" {
    var a = Value(f16).init(10.0);
    var b = Value(f16).init(12.0);
    // forward pass
    var c = a.sub(&b);
    try testing.expectEqual(@as(f16, -2.0), c.data);
    // backward pass
    try c.backward(std.testing.allocator);
    try testing.expectEqual(@as(f16, 1.0), c.grad);
    try testing.expectEqual(@as(f16, 1.0), a.grad);
    try testing.expectEqual(@as(f16, -1.0), b.grad);
}

test "value mul" {
    var a = Value(f16).init(10.0);
    var b = Value(f16).init(12.0);
    // forward pass
    var c = a.mul(&b);
    try testing.expectEqual(@as(f16, 120.0), c.data);
    // backward pass
    try c.backward(std.testing.allocator);
    try testing.expectEqual(@as(f16, 1.0), c.grad);
    try testing.expectEqual(@as(f16, 12.0), a.grad);
    try testing.expectEqual(@as(f16, 10.0), b.grad);
}

test "value multi-ref 1" {
    var a = Value(f16).init(10.0);
    var b = Value(f16).init(12.0);
    var c = Value(f16).init(3.0);
    // forward pass
    var d = a.mul(&b);
    var e = a.mul(&c);
    var f = d.add(&e);
    try testing.expectEqual(@as(f16, 150.0), f.data);
    // backward pass
    try f.backward(std.testing.allocator);
    try testing.expectEqual(@as(f16, 1.0), f.grad);
    try testing.expectEqual(@as(f16, 1.0), e.grad);
    try testing.expectEqual(@as(f16, 1.0), d.grad);
    try testing.expectEqual(@as(f16, 10.0), c.grad);
    try testing.expectEqual(@as(f16, 10.0), b.grad);
    try testing.expectEqual(@as(f16, 15.0), a.grad);
}

test "value multi-ref 2" {
    var a = Value(f16).init(10.0);
    var b = Value(f16).init(12.0);
    var c = Value(f16).init(3.0);
    // forward pass
    var d = a.add(&b);
    var e = a.sub(&c);
    var f = d.mul(&e);
    try testing.expectEqual(@as(f16, (10.0 - 3.0) * (10.0 + 12.0)), f.data);
    // backward pass
    try f.backward(std.testing.allocator);

    // var graph = try f.toGraphViz(std.testing.allocator);
    // std.debug.print("{s}\n", .{graph.items});
    // graph.deinit();

    try testing.expectEqual(@as(f16, 1.0), f.grad);
    try testing.expectEqual(@as(f16, 22.0), e.grad);
    try testing.expectEqual(@as(f16, 7.0), d.grad);
    try testing.expectEqual(@as(f16, -22.0), c.grad);
    try testing.expectEqual(@as(f16, 7.0), b.grad);
    try testing.expectEqual(@as(f16, 29.0), a.grad);
}

test "value tanh" {
    var a = Value(f16).init(0.5);
    var b = a.tanh();
    try testing.expectEqual(@as(f16, 0.4621582), b.data);
    // backward pass
    try b.backward(std.testing.allocator);
    try testing.expectEqual(@as(f16, 0.7861328), a.grad);
}

