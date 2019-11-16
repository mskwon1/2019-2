var mysql = require('mysql')
var http = require('http')
var pw = require('./pw.js')
var url = require('url');
var path = require('path');
var qs = require('querystring');

var db = mysql.createConnection({
  host : 'localhost',
  user : 'root',
  password : pw.pw,
  database : 'games'
});
db.connect();

http.createServer(function(request, response) {
  var _url = request.url;
  var queryData = url.parse(_url, true).query;
  var pathname = url.parse(_url, true).pathname;

  if(pathname == '/') {
    db.query('SELECT * from card2',function (err, cards) {
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
              <th>ID</th>
              <th>SUIT</th>
              <th>RANK</th>
            </tr>
            `
      for (var i=0;i<cards.length;i++) {
        body += `<tr>
        <td>${cards[i].id}</td>
        <td>${cards[i].suit}</td>
        <td>${cards[i].rank}</td>
        </tr>`
      }

      body += '</table> <a href="/create">create</a><a href ="/delete">delete</a></body> </html>'

      response.writeHead(200);
      response.end(body);
    })
  } else if (pathname == '/create') {
    var body = `
      <!doctype html>
      <html>
      <head>
      <meta charset="utf-8">
      <title>CARD</title>
      </head>
      <body>
        <form action="create_process" method="post">
          <input type="number" name="id">
          <br>
          <input type="text" name="suit">
          <br>
          <input type="text" name="rank">
          <br>
          INSERT <input type="submit">
        </form>
      </body>
      </html>
    `;

    response.writeHead(200);
    response.end(body);
  } else if (pathname == '/create_process') {
    var body = '';
    request.on('data', function(data) {
      body = body + data;
    });
    request.on('end', function() {
      var post = qs.parse(body);
      var id = post.id;
      var suit = post.suit;
      var rank = post.rank;
      db.query(`INSERT INTO card2 VALUES(${id},?,?)`, [suit, rank], function(err, result) {
        if (err) {
          throw err;
        }

        response.writeHead(302, {Location:'/'});
        response.end();
      })
    })
  } else if (pathname == '/delete') {
    var body = `
      <!doctype html>
      <html>
      <head>
      <meta charset="utf-8">
      <title>CARD</title>
      </head>
      <body>
        <form action="delete_process" method="post">
          <input type="number" name="id">
          <br>
          DELETE <input type="submit">
        </form>
      </body>
      </html>
    `;

    response.writeHead(200);
    response.end(body);
  } else if (pathname == '/delete_process') {
    var body = '';
    request.on('data', function(data) {
      body = body + data;
    })
    request.on('end', function() {
      var post = qs.parse(body);
      var id = post.id;
      db.query(`DELETE FROM card2 WHERE id = ?`, [id], function(err, result) {
        if (err) {
          throw err;
        }
        response.writeHead(302, {Location :'/'});
        response.end();
      })
    })
  }

}).listen(3000);
