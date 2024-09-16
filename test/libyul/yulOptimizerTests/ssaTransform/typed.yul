{
  let b := true
  let c := false
  c := b
  b := false

  let a := 1
  a := add(a, 1)
  if c {
    a := add(a, 1)
  }
  a := add(a, 1)
  mstore(a, 1)
}
// ----
// step: ssaTransform
//
// {
//     let b_1 := true
//     let b := b_1
//     let c_2 := false
//     let c := c_2
//     let c_3 := b_1
//     c := c_3
//     let b_4 := false
//     b := b_4
//     let a_5 := 1
//     let a := a_5
//     let a_6 := add(a_5, 1)
//     a := a_6
//     if c_3
//     {
//         let a_7 := add(a_6, 1)
//         a := a_7
//     }
//     let a_9 := a
//     let a_8 := add(a_9, 1)
//     a := a_8
//     mstore(a_8, 1)
// }
