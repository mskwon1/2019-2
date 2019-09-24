var fs = require('fs');

fs.readFile('./fs_readFile.js', function (err, data) {
  if (err) {
    throw err;
  }

  console.log(data.toString());
});
