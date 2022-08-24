const std = @import("std");
const testing = std.testing;
const engine = @import("./main.zig");
const tac = testing.allocator;
const Value = engine.Value;

pub fn Neuron(comptime T: type) type {
    return struct {
        const Self = @This();

        b_and_ws: std.ArrayList(Value(T)),
        _signals: std.ArrayList(Value(T)),
        _pre_bias: Value(T),
        _res: Value(T),
        non_linear_op: engine.NonLinearOp, 

        fn initHelper(
            alloc: std.mem.Allocator,
            n_in: usize,
            non_linear_op: engine.NonLinearOp,
        ) std.mem.Allocator.Error!Self {
            var self = Self{
                .b_and_ws = try std.ArrayList(Value(T)).initCapacity(alloc, n_in+1),
                ._signals = try std.ArrayList(Value(T)).initCapacity(alloc, n_in),
                .non_linear_op = non_linear_op,
                ._res = try Value(T).init(alloc, 0),
                ._pre_bias = try Value(T).init(alloc, 0),
            };
            var i : usize = 0;
            while (i < n_in) : (i += 1) {
                try self._signals.append(try Value(T).init(alloc, 0));
            }
            return self;
        }

        pub fn initRand(
            alloc: std.mem.Allocator, 
            rng: std.rand.Random, 
            n_in: usize, 
            non_linear_op: engine.NonLinearOp,
        ) std.mem.Allocator.Error!Self {
            var self = try initHelper(alloc, n_in, non_linear_op);
            var i : usize = 0;
            while (i < n_in + 1) : (i += 1) {
                try self.b_and_ws.append(try Value(T).init(alloc, @floatCast(T, rng.float(f32))));
            }
            return self;
        }

        pub fn initAll(
            alloc: std.mem.Allocator, 
            n_in: usize, 
            num: T,
            non_linear_op: engine.NonLinearOp,
        ) std.mem.Allocator.Error!Self {
            var self = try initHelper(alloc, n_in, non_linear_op);
            var i : usize = 0;
            while (i < n_in + 1) : (i += 1) {
                try self.b_and_ws.append(try Value(T).init(alloc, num));
            }
            return self;
        }

        pub fn deinit(self: *Self) void {
            for (self.b_and_ws.items) |*w| {
                w.deinit();
            }
            for (self._signals.items) |*w| {
                w.deinit();
            }
            self.b_and_ws.deinit();
            self._signals.deinit();
            self._pre_bias.deinit();
            self._res.deinit();
        }

        pub fn params(self: *Self) []Value(T) {
            return &self.b_and_ws;
        }

        fn zeroGrads(self: *Self) void {
            for (self.b_and_ws.items) |*w| {
                w.grad = 0;
            }
            for (self._signals.items) |*s| {
                s.grad = 0;
            }
            self._pre_bias.grad = 0;
            self._res.grad = 0;
        }

        pub fn writeGraphVizInternals(self: *Self, str_builder: *std.ArrayList(u8), cluster: bool) !void {
            var writer = str_builder.writer();
            if (cluster) {
                _ = try writer.print("\t\nsubgraph \"cluster_{*}\" {{\n\tstyle=filled; color=lightgray;\n", .{self});
            }
            for (self.b_and_ws.items) |*w| {
                try w.writeGraphVizInternals(str_builder, cluster);
            }
            for (self._signals.items) |*w| {
                try w.writeGraphVizInternals(str_builder, cluster);
            }
            try self._pre_bias.writeGraphVizInternals(str_builder, cluster);
            try self._res.writeGraphVizInternals(str_builder, cluster);
            if (cluster) {
                _ = try writer.write("}\n");
            }
            _ = writer;
        }

        pub fn run(self: *Self, comptime InputT: type, inputs: []InputT, out: *Value(T)) !void {
            const n_in = self._signals.items.len;
            std.debug.assert(inputs.len == n_in);
            self.zeroGrads();
            var i : usize = 0;
            while (i < n_in) : (i += 1) {
                if (InputT == Value(T)) {
                    self._signals.items[i].setMul(&self.b_and_ws.items[i+1], &inputs[i]);
                } else if (InputT == T) {
                    self._signals.items[i].setMulV(&self.b_and_ws.items[i+1], inputs[i]);
                } else {
                    @compileError("InputT must be type T or Value(T)");
                }
            }
            try self._pre_bias.setAddAll(self._signals.items);
            self._res.setAdd(&self._pre_bias, &self.b_and_ws.items[0]);

            switch (self.non_linear_op) {
               .tanh => out.setTanh(&self._res), 
               .relu => out.setRelu(&self._res), 
               .none => out.setAddV(&self._res, 0),
            }
        }
    };
}

pub fn Layer(comptime T: type) type {
    return struct {
        const Self = @This();
        neurons: std.ArrayList(Neuron(T)),
        _out: std.ArrayList(Value(T)),

        pub fn init(alloc: std.mem.Allocator, rng: std.rand.Random, n_in: usize, n_out: usize) !Self {
            var self = Self{ 
                .neurons = try std.ArrayList(Neuron(T)).initCapacity(alloc, n_out),
                ._out = try std.ArrayList(Value(T)).initCapacity(alloc, n_out),
            };
            var i : usize = 0; 
            while (i < n_out) : (i += 1) {
                try self.neurons.append(try Neuron(T).initRand(alloc, rng, n_in, .tanh));
                try self._out.append(try Value(T).init(alloc, 0));
            }
            return self;
        }

        pub fn deinit(self: *Self) void {
            for (self.neurons.items) |*n| {
                n.deinit();
            }
            self.neurons.deinit();
            for (self._out.items) |*v| {
                v.deinit();
            }
            self._out.deinit();
        }

        pub fn writeGraphVizInternals(self: *Self, str_builder: *std.ArrayList(u8), cluster: bool) !void {
            var writer = str_builder.writer();
            if (cluster) {
                _ = try writer.print("\nsubgraph \"cluster_{*}\" {{\n", .{self});
            }
            for (self.neurons.items) |*n| {
                try n.writeGraphVizInternals(str_builder, cluster);
            }
            if (cluster) {
                _ = try writer.write("}\n");
            }
            _ = try writer.print("\nsubgraph \"cluster_{*}\" {{\n", .{&self._out});
            _ = try writer.write("margin=\"20\"\n");
            for (self._out.items) |*v| {
                try v.writeGraphVizInternals(str_builder, cluster);
            }
            _ = try writer.write("}\n");
        }

        pub fn run(self: *Self, comptime InputT: type, inputs: []InputT) ![]Value(T) {
            var i : usize = 0;
            while (i < self.neurons.items.len) : (i += 1) {
                try self.neurons.items[i].run(InputT, inputs, &self._out.items[i]);
            }
            return self._out.items;
        }

        pub fn backward(self: *Self, alloc: std.mem.Allocator) !void {
            for (self._out.items) |*o| {
                try o.backward(alloc);
            }
        }

        pub fn params(alloc: std.mem.Allocator, self: *Self) !std.ArrayList(Value(T)) {
            var ps = std.ArrayList(Value(T)).init(alloc);
            for (self.neurons.items) |*n| {
                try ps.appendSlice(n.params());
            }
            return ps;
        }
    };
}

pub fn MLP(comptime T: type) type {
    return struct {
        const Self = @This();

        layers: std.ArrayList(Layer(T)),

        pub fn init(alloc: std.mem.Allocator, rng: std.rand.Random, sizes: []const usize) !Self {
            var layers = try std.ArrayList(Layer(T)).initCapacity(alloc, sizes.len);
            for (sizes[0..sizes.len-1]) |sz, i| {
                try layers.append(try Layer(T).init(alloc, rng, sz, sizes[i+1]));
            }
            return Self{ .layers = layers };
        }

        pub fn deinit(self: *Self) void {
            for (self.layers.items) |*layer| {
                layer.deinit();
            }
            self.layers.deinit();
        }

        pub fn writeGraphVizInternals(self: *Self, str_builder: *std.ArrayList(u8), cluster: bool) !void {
            for (self.layers.items) |*l| {
                try l.writeGraphVizInternals(str_builder, cluster);
            }
        }

        pub fn toGraphViz(self: *Self, alloc: std.mem.Allocator, cluster: bool) !std.ArrayList(u8) {
            var str_builder = std.ArrayList(u8).init(alloc);
            var writer = str_builder.writer();
            _ = try writer.write("\ndigraph nodes {\n\tnode [shape=record];\n");
            _ = try writer.write("graph [nodesep=\"0.1\", ranksep=\"2\"];");
            // self contained cluster to keep input outside
            _ = try writer.write("\tsubgraph cluster_1 {input [label=\"input\"];}\n");

            try self.writeGraphVizInternals(&str_builder, cluster);

            _ = try writer.write("}\n");
            return str_builder;
        }

        pub fn backward(self: *Self, alloc: std.mem.Allocator) !void {
            for (self.layers.items) |*l| {
                try l.backward(alloc);
            }
        }

        pub fn run(self: *Self, input: []T) ![]Value(T) {
            var vals : []Value(T) = try self.layers.items[0].run(T, input);
            for (self.layers.items[1..]) |*layer| {
                vals = try layer.run(Value(T), vals);
            }
            return vals;
        }
    };
}
//
// test "neuron run" {
//     // var prng = std.rand.DefaultPrng.init(1);
//     // var rand = prng.random();
//     var neuron = try Neuron(f32).initAll(tac, 5, 1, .relu);
//     defer neuron.deinit();
//     var x : [5]Value(f32) = .{
//         try Value(f32).init(tac, 1),
//         try Value(f32).init(tac, 2),
//         try Value(f32).init(tac, 3),
//         try Value(f32).init(tac, 4),
//         try Value(f32).init(tac, 5),
//     };
//     var out = try Value(f32).init(tac, 0);
//     defer out.deinit();
//     try neuron.run(&x, &out);
//     // try neuron.run(&x, &out);
//     try out.backward(tac);
//     {
//         var graph = try out.toGraphViz(tac);
//         defer graph.deinit();
//         std.debug.print("\n{s}\n", .{graph.items});
//     }
//     for (x[0..]) |*v| {
//         v.deinit();
//     }
// }

test "mlp run" {
    var prng = std.rand.DefaultPrng.init(1);
    var rand = prng.random();
    var sizes : []const usize = &.{8, 16, 4};
    var mlp = try MLP(f32).init(tac, rand, sizes);
    defer mlp.deinit();
    var x = [_]f32{1} ** 8;
    var outs = try mlp.run(&x);
    outs = try mlp.run(&x);
    _ = outs;
    try mlp.backward(tac);
    var graph = try mlp.toGraphViz(tac, true);
    defer graph.deinit();
    std.debug.print("\n{s}\n", .{graph.items});
}
