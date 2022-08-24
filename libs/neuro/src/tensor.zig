const std = @import("std");
const List = std.ArrayList;

pub fn Tensor(comptime T: type) type {
  return struct {
    const Self = @This();

    data: std.ArrayList(Value(T)),

    // pub fn sum(){
    //
    // }
    //
    // pub fn prod(){
    //
    // }

  };
}
