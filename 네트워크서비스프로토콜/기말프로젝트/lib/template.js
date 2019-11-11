module.exports = {
  //<link rel='stylesheet' type='text/css' href='/css/style.css'>
  HTML:function(schedule_list,review_list, body, control){
    return `
    <!doctype html>
    <html>
    <head>
      <link rel='stylesheet' href='/css?css=style.css'>
      <title>나의 여행 다이어리</title>
      <meta charset = "utf-8">
    </head>
    <body>
      <div class = "sidenav">
        <div class = "submenu">여행일정공유
          ${schedule_list}
          <a id='new_schedule' href="/schedule_create">새로만들기</a>
        </div>
        <div class = "submenu">여행후기
          ${review_list}
          <a id='new_review' href="/review_create">새로쓰기</a>
        </div>
      </div>
      <div class = "main">
        <h1 align = 'center' id='main_title'><a href="/">나의 여행 다이어리</a></h1>
        ${control}
        ${body}
      </div>
    </body>
    </html>
    `;
  },schedule_list:function(schedules){
    var list = '';

    var i = 0;
    while(i < schedules.length){
      list = list + `<a href="/schedule?id=${schedules[i].SCHEDULE_ID}">${schedules[i].SCHEDULE_NAME}</a>`;
      i = i + 1;
    }

    return list;
  },review_list:function(schedules){
    var list = '';

    var i = 0;
    while(i < schedules.length){
      list = list + `<a href="/review?id=${schedules[i].SCHEDULE_ID}">${schedules[i].SCHEDULE_NAME}</a>`;
      i = i + 1;
    }

    return list;
  }
}
