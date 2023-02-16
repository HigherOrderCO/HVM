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
  pub fn new() -> U64Map<A> {
    // std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::default());
    U64Map { data: Vec::new() }
  }

  pub fn from_hashmap(old_map: &mut std::collections::HashMap<u64, A>) -> U64Map<A> {
    let mut new_map: U64Map<A> = U64Map::new();
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

  pub fn insert(&mut self, key: u64, val: A) {
    while self.data.len() <= key as usize {
      self.data.push(None);
    }
    self.data[key as usize] = Some(val);
  }

  pub fn get(&self, key: &u64) -> Option<&A> {
    if let Some(Some(got)) = self.data.get(*key as usize) {
      return Some(got);
    }
    None
  }
}
