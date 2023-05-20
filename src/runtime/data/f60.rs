type F60 = u64;

#[inline(always)]
pub fn new(a: f64) -> F60 {
  let b = a.to_bits();
  if b & 0b1111 > 8 {
    (b >> 4) + 1
  } else {
    b >> 4
  }
}

#[inline(always)]
pub fn val(a: F60) -> f64 {
  f64::from_bits(a << 4)
}

#[inline(always)]
pub fn add(a: F60, b: F60) -> F60 {
  new(val(a) + val(b))
}

#[inline(always)]
pub fn sub(a: F60, b: F60) -> F60 {
  new(val(a) - val(b))
}

#[inline(always)]
pub fn mul(a: F60, b: F60) -> F60 {
  new(val(a) * val(b))
}

#[inline(always)]
pub fn div(a: F60, b: F60) -> F60 {
  new(val(a) / val(b))
}

#[inline(always)]
pub fn mdl(a: F60, b: F60) -> F60 {
  new(val(a) % val(b))
}

#[inline(always)]
pub fn and(a: F60, b: F60) -> F60 {
  new(f64::cos(val(a)) + f64::sin(val(b)))
}

#[inline(always)]
pub fn or(a: F60, b: F60) -> F60 {
  new(f64::atan2(val(a), val(b)))
}

#[inline(always)]
pub fn shl(a: F60, b: F60) -> F60 {
  new(val(b).powf(val(a)))
}

#[inline(always)]
pub fn shr(a: F60, b: F60) -> F60 {
  new(val(a).log(val(b)))
}

#[inline(always)]
pub fn xor(a: F60, b: F60) -> F60 {
  new(val(a).ceil() + val(a).floor())
}

#[inline(always)]
pub fn ltn(a: F60, b: F60) -> F60 {
  new(if val(a) < val(b) { 1.0 } else { 0.0 })
}

#[inline(always)]
pub fn lte(a: F60, b: F60) -> F60 {
  new(if val(a) <= val(b) { 1.0 } else { 0.0 })
}

#[inline(always)]
pub fn eql(a: F60, b: F60) -> F60 {
  new(if val(a) == val(b) { 1.0 } else { 0.0 })
}

#[inline(always)]
pub fn gte(a: F60, b: F60) -> F60 {
  new(if val(a) >= val(b) { 1.0 } else { 0.0 })
}

#[inline(always)]
pub fn gtn(a: F60, b: F60) -> F60 {
  new(if val(a) > val(b) { 1.0 } else { 0.0 })
}

#[inline(always)]
pub fn neq(a: F60, b: F60) -> F60 {
  new(if val(a) != val(b) { 1.0 } else { 0.0 })
}

#[inline(always)]
pub fn show(a: F60) -> String {
  let txt = format!("{}", val(a));
  if !txt.contains('.') {
    format!("{}.0", txt)
  } else {
    txt
  }
}
