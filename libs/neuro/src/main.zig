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

pub fn Backward(comptime T: type) type {
    return struct {
        const Self = @This();

        left: *Value(T),
        right: ?*Value(T) = null,

        pub fn backward(self: *Self, op: Op, out: *const Value(T)) void {
            switch (op) {
                .add => {
                    self.left.grad += out.grad;            
                    self.right.?.grad += out.grad;
                },
                .sub => {
                    self.left.grad += out.grad;            
                    self.right.?.grad -= out.grad;
                },
                .mul => {
                    self.left.grad += self.right.?.data * out.grad;            
                    self.right.?.grad += self.left.data * out.grad;
                },
                .none => {},
                else => @panic("Unimplemented backward operation"),
            }
        }
    };
}

pub fn Value(comptime T: type) type {
    return struct {
        const Self = @This();

        data: T,
        grad: T = 0,
        // The operation that created this value
        op: Op = .none,
        _prev: [2]?*Self = .{null} ** 2,
        _backward: Backward(T),

        pub fn init(value: T) Self {
            var self = Self{
                .data = value, 
                ._backward = undefined,
            }; 
            self._backward = Backward(T){ .left = &self };
            return self;
        }

        pub fn add(self: *Self, other: *Self) Self {
            return Self{
                .data = self.data + other.data,
                .op = .add,
                ._backward = Backward(T){
                    .left = self,
                    .right = other,
                },
                ._prev = .{self, other}
            }; 
        }

        pub fn sub(self: *Self, other: *Self) Self {
            return Self{
                .data = self.data - other.data,
                .op = .sub,
                ._backward = Backward(T){
                    .left = self,
                    .right = other,
                },
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
                ._backward = Backward(T){
                    .left = self,
                    .right = other,
                },
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
                node._backward.backward(node.op, node);
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
    try testing.expectEqual(@as(f16, 1.0), f.grad);
    try testing.expectEqual(@as(f16, 22.0), e.grad);
    try testing.expectEqual(@as(f16, 7.0), d.grad);
    try testing.expectEqual(@as(f16, -22.0), c.grad);
    try testing.expectEqual(@as(f16, 7.0), b.grad);
    try testing.expectEqual(@as(f16, 29.0), a.grad);
}

