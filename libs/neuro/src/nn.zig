const std = @import("std");
const testing = std.testing;
const engine = @import("./main.zig");
const Value = engine.Value;

pub fn Neuron(comptime T: type) type {
    return struct {
        const Self = @This();

        b_and_ws: std.ArrayList(Value(T)),
        // stores intermediate values (instead of allocating)
        _totals: std.ArrayList(Value(T)),
        _in_vals: std.ArrayList(Value(T)),
        non_linear_op: engine.NonLinearOp, 

        pub fn initRand(
            alloc: std.mem.Allocator, 
            rng: std.rand.Random, 
            n_in: usize, 
            non_linear_op: engine.NonLinearOp,
        ) std.mem.Allocator.Error!Self {
            var self = Self{
                .b_and_ws = std.ArrayList(Value(T)).init(alloc),
                ._totals = std.ArrayList(Value(T)).init(alloc),
                ._in_vals = std.ArrayList(Value(T)).init(alloc),
                .non_linear_op = non_linear_op,
            };
            try self.b_and_ws.resize(n_in+1);
            try self._totals.resize(n_in+1);
            try self._in_vals.resize(n_in);

            self.b_and_ws.items[0] = Value(T).init(@floatCast(T, rng.float(f32)));
            for (self.b_and_ws.items[1..]) |*wptr| {
                const rand_float = @floatCast(T, rng.float(f32));
                wptr.* = Value(T).init(rand_float);
            }
            return self;
        }
        pub fn deinit(self: *Self) void {
            self.b_and_ws.deinit();
            self._totals.deinit();
            self._in_vals.deinit();
        }

        pub fn params(self: *Self) []Value(T) {
            return &self.b_and_ws;
        }

        pub fn runV(self: *Self, inputs: []const T) Value(T) {
            const n_in = self._in_vals.items.len;
            std.debug.assert(inputs.len == n_in);
            self._totals.items[0] = self.b_and_ws.items[0];
            for (self.b_and_ws.items[1..]) |*wval, i| {
               self._in_vals.items[i] = wval.mulV(inputs[i]);
               self._totals.items[i+1] = self._totals.items[i].add(&self._in_vals.items[i]);
            }
            return switch (self.non_linear_op) {
               .tanh => self._totals.items[n_in].tanh(), 
               .relu => self._totals.items[n_in].relu(), 
               .none => self._totals.items[n_in],
            };
        }
        pub fn run(self: *Self, inputs: []Value(T)) Value(T) {
            const n_in = self._in_vals.items.len;
            std.debug.assert(inputs.len == n_in);
            self._totals.items[0] = self.b_and_ws.items[0];
            for (self.b_and_ws.items[1..]) |*wval, i| {
               self._in_vals.items[i] = wval.mul(&inputs[i]);
               self._totals.items[i+1] = self._totals.items[i].add(&self._in_vals.items[i]);
            }
            return switch (self.non_linear_op) {
               .tanh => self._totals.items[n_in].tanh(), 
               .relu => self._totals.items[n_in].relu(), 
               .none => self._totals.items[n_in],
            };
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
            try self.neurons.resize(n_out);
            try self._out.resize(n_out);
            var i : usize = 0; 
            while (i < n_out) : (i += 1) {
                self.neurons.items[i] = try Neuron(T).initRand(alloc, rng, n_in, .tanh);
            }
            return self;
        }

        pub fn deinit(self: *Self) void {
            for (self.neurons.items) |*n| {
                n.deinit();
            }
            self.neurons.deinit();
            self._out.deinit();
        }

        pub fn run(self: *Self, inputs: []Value(T)) []Value(T) {
            for (self.neurons.items) |*n, i| {
                self._out.items[i] = n.run(inputs);
            }
            return self._out.items;
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

        pub fn run(self: *Self, input: []Value(T)) []Value(T) {
            var vals : []Value(T) = input;
            for (self.layers.items) |*layer| {
                vals = layer.run(vals);
            }
            return vals;
        }
    };
}
//
// test "neuron fire" {
//     var prng = std.rand.DefaultPrng.init(1);
//     var rand = prng.random();
//     var n = try Neuron(f32).initRand(std.testing.allocator, rand, 5, .tanh);
//     defer n.deinit();
//     var res = n.run(&.{1.0, 2.0, 3.0, 4.0, 5.0});
//     try res.backward(std.testing.allocator);
//     var graph = try res.toGraphViz(std.testing.allocator);
//     defer graph.deinit();
//     std.debug.print("\n{s}\n", .{graph.items});
// }

test "mlp fire" {
    var prng = std.rand.DefaultPrng.init(1);
    var rand = prng.random();
    var sizes : []const usize = &.{3, 4, 4, 1};
    var mlp = try MLP(f32).init(std.testing.allocator, rand, sizes);
    defer mlp.deinit();
    var x = &.{
        Value(f32).init(1), 
        Value(f32).init(2), 
        Value(f32).init(3), 
    };
    var outs = mlp.run(x);
    try outs[0].backward(std.testing.allocator);
    var graph = try outs[0].toGraphViz(std.testing.allocator);
    defer graph.deinit();
    std.debug.print("\n{s}\n", .{graph.items});
}
