var body = '1232131233'
let promise = new Promise(function(resolve, reject) {
  body += testFunc('hello world')
  resolve(body)
})

promise.then(function(contents) {
  console.log(contents);
}, function (err) {
  console.log(err.message)
})

function testFunc(body) {
  for (var i=0;i<50;i++) {
    body += i
  }

  return body
}
