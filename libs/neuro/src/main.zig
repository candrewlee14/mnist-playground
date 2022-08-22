const std = @import("std");
const testing = std.testing;

const Op = enum(u8) {
    add,
    mul,
    sub,
    div,
    pow,
    relu,
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
                node.backwardHandler(node.op, node);
            }
        }

        fn backwardHandler(self: *Self, op: Op, out: *const Value(T)) void {
            var left = self._prev[0];
            var right = self._prev[1];
            switch (op) {
                .add => {
                    left.?.grad += out.grad;            
                    right.?.grad += out.grad;
                },
                .sub => {
                    left.?.grad += out.grad;            
                    right.?.grad -= out.grad;
                },
                .mul => {
                    left.?.grad += right.?.data * out.grad;            
                    right.?.grad += left.?.data * out.grad;
                },
                .none => {},
                else => @panic("Unimplemented backward operation"),
            }
        }
    };
}

test "value init" {
    var val = Value(f16).init(10.0);
    std.debug.print("Sizeof Value(f16): {}\n", .{@sizeOf(Value(f16))});
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
    try testing.expectEqual(@as(f16, 1.0), f.grad);
    try testing.expectEqual(@as(f16, 22.0), e.grad);
    try testing.expectEqual(@as(f16, 7.0), d.grad);
    try testing.expectEqual(@as(f16, -22.0), c.grad);
    try testing.expectEqual(@as(f16, 7.0), b.grad);
    try testing.expectEqual(@as(f16, 29.0), a.grad);
}

