function Fn(x: number): number {
  return x == 0 ? 1 : Fn(x - 1) + Fn(x - 1);
}

function Main() {
  return [Fn(26), Fn(26), Fn(26), Fn(26), Fn(26), Fn(26), Fn(26), Fn(26)];
}

console.log(Main());
