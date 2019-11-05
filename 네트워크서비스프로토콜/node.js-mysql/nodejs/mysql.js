var mysql = require('mysql');
var pw = require('../pw.js')

var connection = mysql.createConnection({
  host : 'localhost',
  user : 'root',
  password : pw.pw,
  database : 'opentutorials'
});

connection.connect();
connection.query('SELECT * FROM topic', function(error, results, fileds) {
  if (error) {
    console.log(error);
  }

  console.log(results);
});

connection.end();
