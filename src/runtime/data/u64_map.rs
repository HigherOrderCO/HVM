// std::collections::HashMap<u64, A, std::hash::BuildHasherDefault<nohash_hasher::NoHashHasher<u64>>>;

pub struct U64Map<A> {
  pub data: Vec<Option<A>>,
}

impl<A> Default for U64Map<A> {
  fn default() -> Self {
    Self::new()
  }
}

impl<A> U64Map<A> {
  pub fn new() -> Self {
    // std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::default());
    Self { data: vec![] }
  }

  pub fn from_hashmap(old_map: &mut std::collections::HashMap<u64, A>) -> Self {
    let mut new_map = Self::new();
    for (key, val) in old_map.drain() {
      new_map.insert(key, val);
    }
    new_map
  }

  pub fn push(&mut self, val: A) -> u64 {
    let key = self.data.len() as u64;
    self.insert(key, val);
    key
  }

  pub fn insert(&mut self, key: impl Into<u64>, val: A) {
    let k = key.into() as usize;
    while self.data.len() <= k as usize {
      self.data.push(None);
    }
    self.data[k] = Some(val);
  }

  pub fn get(&self, key: impl Into<u64>) -> Option<&A> {
    if let Some(Some(got)) = self.data.get(key.into() as usize) {
      return Some(got);
    }
    None
  }
}
