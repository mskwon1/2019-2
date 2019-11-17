var mysql = require('mysql')
var http = require('http')

var db = mysql.createConnection({
  host : 'localhost',
  user : 'root',
  password : /*니 패스워드 */,
  database : 'games'
});
db.connect();

http.createServer(function(request, response) {
  db.query('SELECT * from card',function (err, cards) {
    console.log(cards);
    var body = `
    <!doctype html>
      <html>
        <head>
        <title>CARD</title>
        </head>
        <body>
        <table border = "1">
          <tr>
            <th>SUIT</th>
            <th>RANK</th>
          </tr>
          `
    for (var i=0;i<cards.length;i++) {
      body += `<tr>
      <td>${cards[i].suit}</td>
      <td>${cards[i].rank}</td>
      </tr>`
    }

    body += '</table> </body> </html>'

    response.writeHead(200);
    response.end(body);
  })
}).listen(3000);
