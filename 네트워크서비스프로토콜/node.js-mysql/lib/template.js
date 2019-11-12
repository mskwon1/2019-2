module.exports = {
  HTML:function(title, list, body, control){
    return `
    <!doctype html>
    <html>
    <head>
      <title>WEB1 - ${title}</title>
      <meta charset="utf-8">
    </head>
    <body>
      <h1><a href="/">WEB</a></h1>
      ${list}
      ${control}
      ${body}
    </body>
    </html>
    `;
  },
  list:function(filelist){
    var list = '<ul>';
    var i = 0;
    while(i < filelist.length){
      list = list + `<li><a href="/?id=${filelist[i].id}">${filelist[i].title}</a></li>`;
      i = i + 1;
    }
    list = list+'</ul>';
    return list;
  },
  authorSelect:function(authors) {
    var tag='';
    tag += '<select name="author">'
    for (var i=0;i<authors.length;i++) {
      tag += `<option value="${authors[i].id}">${authors[i].name}</option>`
    }
    tag += '</select>'

    return tag
  }
}
