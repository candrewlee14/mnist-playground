const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const tac = testing.allocator;

pub const NonLinearOp = enum {
    relu,   
    tanh,   
    none,
};

pub const Op = enum {
    add,
    mul,
    mulv,
    sub,
    div,
    pow,
    // non-linear
    relu,
    tanh,
    // no op
    none,

    pub fn toStr(self: Op) []const u8 {
        return switch (self) {
            .add => "+",
            .sub => "-",
            .mul => "x",
            .mulv => "x input",
            .div => "/",
            .pow => "^",
            .relu => "ReLU",
            .tanh => "tanh",
            .none => "=",
        };
    }
};

pub fn Value(comptime T: type) type {
    return struct {
        const Self = @This();

        data: T,
        grad: T = 0,
        // The operation that created this value
        op: Op = .none,
        _prev: std.ArrayList(*Self),

        pub fn init(alloc: Allocator, value: T) !Self {
            var self = Self{
                .data = value, 
                ._prev = try std.ArrayList(*Self).initCapacity(alloc, 2),
            }; 
            return self;
        }
        pub fn deinit(self: *Self) void {
            self._prev.deinit();
        }

        pub fn setAddAll(self: *Self, vals: []Value(T)) !void {
            self.op = .add;
            const not_initted = self._prev.items.len == 0;
            for (vals) |*v| {
                self.data += v.data;
                if (not_initted) {
                    try self._prev.append(v);
                }
            }
        }

        pub fn setMulAll(self: *Self, vals: []Value(T)) !void {
            self.op = .mul;
            const not_initted = self._prev.items.len == 0;
            for (vals) |*v| {
                self.data *= v.data;
                if (not_initted) {
                    try self._prev.append(v);
                }
            }
        }
        pub fn setAdd(self: *Self, one: *Self, two: *Self) void {
            if (self._prev.items.len == 0) {
                self._prev.appendAssumeCapacity(one); 
                self._prev.appendAssumeCapacity(two);
            }
            self.data = one.data + two.data;
            self.op = .add;
        }
        pub fn setAddV(self: *Self, one: *Self, two: T) void {
            if (self._prev.items.len == 0) {
                self._prev.appendAssumeCapacity(one); 
            }
            self.data = one.data + two;
            self.op = .add;
        }
        pub fn setSub(self: *Self, one: *Self, two: *Self) void {
            if (self._prev.items.len == 0) {
                self._prev.appendAssumeCapacity(one); 
                self._prev.appendAssumeCapacity(two);
            }
            self.data = one.data - two.data;
            self.op = .sub;
        }

        pub fn setMul(self: *Self, one: *Self, two: *Self) void {
            if (self._prev.items.len == 0) {
                self._prev.appendAssumeCapacity(one); 
                self._prev.appendAssumeCapacity(two);
            }
            self.data = one.data * two.data;
            self.op = .mul;
        }

        pub fn setMulV(self: *Self, one: *Self, two: T) void {
            if (self._prev.items.len == 0) {
                self._prev.appendAssumeCapacity(one); 
            }
            self.data = one.data * two;
            self.op = .mulv;
        }

        pub fn setRelu(self: *Self, one: *Self) void {
            if (self._prev.items.len == 0) {
                self._prev.appendAssumeCapacity(one); 
            }
            self.data = if (one.data > 0) one.data else 0;
            self.op = .relu;
        }

        pub fn setTanh(self: *Self, one: *Self) void {
            if (self._prev.items.len == 0) {
                self._prev.appendAssumeCapacity(one); 
            }
            const exp2 = @exp(one.data * 2);
            self.data = (exp2 - 1)/(exp2 + 1);
            self.op = .tanh;
        }


        fn buildTopo(
            self: *Self, 
            topo: *std.ArrayList(*Self),
            visited: *std.AutoArrayHashMap(*Self, void),
        ) std.mem.Allocator.Error!void {
            if (visited.get(self) == null) {
                try visited.put(self, .{});
                for (self._prev.items) |child| {
                    try child.buildTopo(topo, visited);
                }
                try topo.append(self);
            }
        }

        pub fn writeGraphVizInternals(self: *Self, str_builder: *std.ArrayList(u8), cluster: bool) !void {
            // var writer = str_builder.writer();
            // _ = try writer.print("\t\"{*}\" [label=\"data={d}|grad={d}\"];\n", 
            //     .{self, self.data, self.grad});
            // if (self.op != .none) {
            //     _ = try writer.print("\t\"{*}-op\" [shape=ellipse, label=\"{s}\"];\n",
            //         .{self, self.op.toStr()});
            //     _ = try writer.print("\t\"{0*}-op\" -> \"{0*}\";\n", .{self});
            // }
            // if (self.op == .mulv) {
            //     _ = try writer.print("\tinput -> \"{0*}-op\";\n", .{self});
            // }
            // for (self._prev.items) |child| {
            //     _ = try writer.print("\t\"{*}\" -> \"{*}-op\"\n", .{child, self});
            // }
            _ = cluster;
            var writer = str_builder.writer();
            // _ = try writer.print("\t\"{*}\" [label=\"{s}|{{data={d}|grad={d} }}\"];\n", 
            //     .{self, self.op.toStr(),self.data, self.grad});
            _ = try writer.print("\t\"{*}\" [label=\"{s}\"];\n", 
                 .{self, self.op.toStr()});
            if (self.op == .mulv) {
                _ = try writer.print("\tinput -> \"{0*}\";\n", .{self});
            }
            for (self._prev.items) |child| {
                _ = try writer.print("\t\"{*}\" -> \"{*}\"\n", .{child, self});
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
                try node.writeGraphVizInternals(&str_builder);
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
            switch (op) {
                .add => {
                    for (self._prev.items) |n| {
                        n.grad += self.grad;
                    }
                },
                .sub => {
                    var left = self._prev.items[0];
                    var right = self._prev.items[1];
                    left.grad += self.grad;            
                    right.grad -= self.grad;
                },
                .mul => {
                    for (self._prev.items) |n| {
                        n.grad += (self.data / n.data * self.grad);
                    }
                },
                .mulv => {
                    var left = self._prev.items[0];
                    left.grad += self.data/left.data * self.grad;            
                },
                .relu => {
                    var left = self._prev.items[0];
                    left.grad += if (self.data > 0) self.grad else 0;
                },
                .tanh => {
                    var left = self._prev.items[0];
                    left.grad += (1 - (self.data * self.data)) * self.grad;
                },
                .none => {},
                else => @panic("Unimplemented backward operation"),
            }
        }
    };
}

test "value init" {
    var val = try Value(f16).init(tac, 10.0);
    defer val.deinit();
    try testing.expectEqual(@as(f16, 10.0), val.data);
    try testing.expectEqual(@as(f16, 0), val.grad);
}

test "value add" {
    var a = try Value(f16).init(tac, 10.0);
    defer a.deinit();
    var b = try Value(f16).init(tac, 12.0);
    defer b.deinit();
    // forward pass
    var c = try Value(f16).init(tac, 0.0);
    defer c.deinit();
    c.setAdd(&a, &b);
    try testing.expectEqual(@as(f16, 22.0), c.data);
    // backward pass
    try c.backward(tac);
    try testing.expectEqual(@as(f16, 1.0), c.grad);
    try testing.expectEqual(@as(f16, 1.0), a.grad);
    try testing.expectEqual(@as(f16, 1.0), b.grad);
}

test "value sub" {
    var a = try Value(f16).init(tac, 10.0);
    defer a.deinit();
    var b = try Value(f16).init(tac, 12.0);
    defer b.deinit();
    // forward pass
    var c = try Value(f16).init(tac, 0.0);
    defer c.deinit();
    c.setSub(&a, &b);
    try testing.expectEqual(@as(f16, -2.0), c.data);
    // backward pass
    try c.backward(tac);
    try testing.expectEqual(@as(f16, 1.0), c.grad);
    try testing.expectEqual(@as(f16, 1.0), a.grad);
    try testing.expectEqual(@as(f16, -1.0), b.grad);
}

test "value mul" {
    var a = try Value(f16).init(tac, 10.0);
    defer a.deinit();
    var b = try Value(f16).init(tac, 12.0);
    defer b.deinit();
    // forward pass
    var c = try Value(f16).init(tac, 0.0);
    defer c.deinit();
    c.setMul(&a, &b);
    try testing.expectEqual(@as(f16, 120.0), c.data);
    // backward pass
    try c.backward(tac);
    try testing.expectEqual(@as(f16, 1.0), c.grad);
    try testing.expectEqual(@as(f16, 12.0), a.grad);
    try testing.expectEqual(@as(f16, 10.0), b.grad);
}

test "value mulV" {
    var a = try Value(f16).init(tac, 10.0);
    defer a.deinit();
    var b : f16 = 12;
    // forward pass
    var c = try Value(f16).init(tac, 0.0);
    defer c.deinit();
    c.setMulV(&a, b);
    try testing.expectEqual(@as(f16, 120.0), c.data);
    // backward pass
    try c.backward(tac);
    try testing.expectEqual(@as(f16, 1.0), c.grad);
    try testing.expectEqual(@as(f16, 12.0), a.grad);
}

test "value multi-ref 1" {
    var a = try Value(f16).init(tac, 10.0);
    var b = try Value(f16).init(tac, 12.0);
    var c = try Value(f16).init(tac, 3.0);
    defer a.deinit();
    defer b.deinit();
    defer c.deinit();
    var d = try Value(f16).init(tac, 0.0);
    var e = try Value(f16).init(tac, 0.0);
    var f = try Value(f16).init(tac, 0.0);
    defer d.deinit();
    defer e.deinit();
    defer f.deinit();
    // forward pass
    d.setMul(&a, &b);
    e.setMul(&a, &c);
    f.setAdd(&d, &e);
    try testing.expectEqual(@as(f16, 150.0), f.data);
    // backward pass
    try f.backward(tac);
    try testing.expectEqual(@as(f16, 1.0), f.grad);
    try testing.expectEqual(@as(f16, 1.0), e.grad);
    try testing.expectEqual(@as(f16, 1.0), d.grad);
    try testing.expectEqual(@as(f16, 10.0), c.grad);
    try testing.expectEqual(@as(f16, 10.0), b.grad);
    try testing.expectEqual(@as(f16, 15.0), a.grad);
}

test "value multi-ref 2" {
    var a = try Value(f16).init(tac, 10.0);
    var b = try Value(f16).init(tac, 12.0);
    var c = try Value(f16).init(tac, 3.0);
    defer a.deinit();
    defer b.deinit();
    defer c.deinit();
    var d = try Value(f16).init(tac, 0.0);
    var e = try Value(f16).init(tac, 0.0);
    var f = try Value(f16).init(tac, 0.0);
    defer d.deinit();
    defer e.deinit();
    defer f.deinit();
    // forward pass
    d.setAdd(&a, &b);
    e.setSub(&a, &c);
    f.setMul(&d, &e);
    try testing.expectEqual(@as(f16, (10.0 - 3.0) * (10.0 + 12.0)), f.data);
    // backward pass
    try f.backward(tac);

    try testing.expectEqual(@as(f16, 1.0), f.grad);
    try testing.expectEqual(@as(f16, 22.0), e.grad);
    try testing.expectEqual(@as(f16, 7.0), d.grad);
    try testing.expectEqual(@as(f16, -22.0), c.grad);
    try testing.expectEqual(@as(f16, 7.0), b.grad);
    try testing.expectEqual(@as(f16, 29.0), a.grad);
}

test "value tanh" {
    var a = try Value(f16).init(tac, 0.5);
    var b = try Value(f16).init(tac, 0.0);
    b.setTanh(&a);
    defer a.deinit();
    defer b.deinit();
    try testing.expectEqual(@as(f16, 0.4621582), b.data);
    // backward pass
    try b.backward(tac);
    try testing.expectEqual(@as(f16, 0.7861328), a.grad);
}

test "value addAll" {
    var a = try Value(f16).init(tac, 10.0);
    var b = try Value(f16).init(tac, 12.0);
    var c = try Value(f16).init(tac, 3.0);
    var d = try Value(f16).init(tac, 4.0);
    var out = try Value(f16).init(tac, 0.0);
    defer a.deinit();
    defer b.deinit();
    defer c.deinit();
    defer d.deinit();
    defer out.deinit();
    try out.setAddAll(&.{a, b, c, d});
    try testing.expectEqual(@as(f16, 29.0), out.data);
}
