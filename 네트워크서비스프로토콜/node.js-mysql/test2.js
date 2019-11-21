var mysql = require('mysql')
var http = require('http')
var pw = require('./pw.js')
var url = require('url');
var path = require('path');
var qs = require('querystring');
var express = require('express');
var bodyParser = require('body-parser')

var db = mysql.createConnection({
  host : 'localhost',
  user : 'root',
  password : pw.pw,
  database : 'games'
});
db.connect();

var app = express();

app.use(bodyParser.urlencoded({ extended: false }));

app.get('/', function(req, res) {
  db.query('SELECT * from card2',function (err, cards) {
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

    body += '</table> <a href="/create">create</a> <a href ="/delete">delete</a></body> </html>'

    res.send(body);
  })
})

app.get('/create', function(req, res) {
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

  res.send(body);
});

app.post('/create_process', function(req, res) {
  var post = req.body;
  db.query(`INSERT INTO card2 VALUES(${post.id},?,?)`, [post.suit, post.rank], function(err, result) {
    if (err) {
      throw err;
    }

    res.redirect('/');
  })
})

app.get('/delete', function(req, res) {
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

  res.send(body);
})

app.post('/delete_process', function(req, res) {
  var post = req.body;
  db.query(`DELETE FROM card2 WHERE id = ?`, [post.id], function(err, result) {
    if (err) {
      throw err;
    }
    res.redirect('/');
  })
})

app.listen(3000, () => console.log('Server running at port 3000'));

// http.createServer(function(request, response) {
//   var _url = request.url;
//   var queryData = url.parse(_url, true).query;
//   var pathname = url.parse(_url, true).pathname;
//
//   if(pathname == '/') {
//
//   } else if (pathname == '/create') {
//
//   } else if (pathname == '/delete') {

//   } else if (pathname == '/delete_process') {

//   }
//
// }).listen(3000);
