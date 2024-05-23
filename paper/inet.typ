#import "@preview/cetz:0.2.2": draw, canvas

#let port = (name, pos, dir) => {
  import draw: *
  group(name: name, {
    translate(pos)
    rotate(dir)
    // scale(1.5)
    anchor("p", (0, 0))
    anchor("c", (0, 0.5))
  })
}

#let agent = (..agent) => (..args) => {
  import draw: *
  let style = agent.named().at("style", default: ())
  let name = args.named().at("name")
  let pos = args.named().at("pos")
  let rot = args.named().at("rot", default: 0deg)
  group(name: name, {
    translate(pos)
    rotate(rot)
    translate((0, -calc.sqrt(3)/4))
    stroke(2pt)
    line((-.5, 0), (.5, 0), (0, calc.sqrt(3)/2), close: true, ..style, stroke: 0.5pt)
    port("0", (0, calc.sqrt(3)/2), 0deg)
    port("1", (-1/2+1/3, 0), 180deg)
    port("2", (+1/2-1/3, 0), 180deg)
  })
}

#let link = (a, b) => {
  import draw: *
  stroke(2pt)
  bezier(a + ".p", b + ".p", a + ".c", b + ".c", stroke: 0.5pt)
}

#let con = agent()
#let dup = agent(style: (fill: black))
