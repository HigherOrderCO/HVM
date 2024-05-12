#flavor core

// Old Types:
// ----------

// List Type
// (builtin)

// String Type
// (builtin)

// IO Type
// (should be builtin)
data IO
  = (Done term)
  | (PutText   text      cont)
  | (GetText             cont)
  | (WriteFile file data cont)
  | (ReadFile  file      cont)
  | (GetTime             cont)
  | (Sleep     time      cont)
  | (DrawImage tree      cont)

// New Type Conversions
// --------------------
// (shouldn't be needeed)

// Note: I'm not using `_` to make it easier to convert from CamelCase to
// uppercase constructor tag names when generating global defs on Bend

// List Tags
NIL  = 0
CONS = 1

// IO Tags
DONE      = 0
PUTTEXT   = 1
GETTEXT   = 2
WRITEFILE = 3
READFILE  = 4
GETTIME   = 5
SLEEP     = 6
DRAWIMAGE = 7

// Converts old String to new encoding
(MkStr (String/cons x xs)) = λt (t CONS x (MkStr xs))
(MkStr String/nil)         = λt (t NIL)

// Converts old IO to new encoding
(MkIO (IO/Done                term)) = λt(t DONE      term)
(MkIO (IO/PutText   text      cont)) = λt(t PUTTEXT   text      λx(MkIO (cont x)))
(MkIO (IO/GetText             cont)) = λt(t GETTEXT             λx(MkIO (cont x)))
(MkIO (IO/WriteFile file data cont)) = λt(t WRITEFILE file data λx(MkIO (cont x)))
(MkIO (IO/ReadFile  file      cont)) = λt(t READFILE  file      λx(MkIO (cont x)))
(MkIO (IO/GetTime             cont)) = λt(t GETTIME             λx(MkIO (cont x)))
(MkIO (IO/Sleep     time      cont)) = λt(t SLEEP     time      λx(MkIO (cont x)))
(MkIO (IO/DrawImage tree      cont)) = λt(t DRAWIMAGE tree      λx(MkIO (cont x)))

// Main
// ----
// (this should work without the 'Mk' fns)

// IO Example
Main = (MkIO
  (IO/PutText (MkStr "hello\n") λcont
  (IO/PutText (MkStr "world\n") λcont
  (IO/DrawImage ((0xFF0000,0x00FF00),(0x0000FF,0xFF00FF)) λcont // RGBP square
  (IO/Sleep (0x77,0x359400) λcont // 2 seconds
  (IO/Done 42))))))

// IO Monad tests
// --------------
// (works already!)

//// IO Units
//DoPrint = λtext (Print text λx(Done x))

//// IO Monad
//IO/bind = λa (a λat switch at {
  //0: λterm λb (b term)
  //1: λtext λcont λb (Print text λx(IO/bind (cont x) b))
  //_: λb 0
//})
  
//Main =
  //do IO {
    //ask x = (DoPrint (Str "hello"))
    //ask y = (DoPrint (Str "word"))
    //(Done 0)
  //}
