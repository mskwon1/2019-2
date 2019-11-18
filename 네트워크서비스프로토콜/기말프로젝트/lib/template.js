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
          <a id='new_schedule' href="/create_schedule">새로만들기</a>
        </div>
        <div class= "submenu">추가하기
          <a href="/add_place">여행지 추가하기</a>
          <a href="/add_activity">활동 추가하기</a>
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
  },timebox:function(default_time) {
    var html = '';

    var hour, min;
    for (var i=0;i<49;i++) {
      if(i < 20) {
        hour = '0' + parseInt(i/2);
      } else {
        hour = '' + parseInt(i/2);
      }

      if (i % 2 == 0) {
        min = '00'
      } else {
        min = '30'
      }

      time = hour + ':' + min + ':00';

      if (default_time == time) {
        html += `<option value=${time} selected="selected">${time}</option>`
      } else {
        html += `<option value=${time}>${time}</option>`
      }
    }

    return html
  },placeCombobox:function(values) {
    var html = ''
    for (var i=0;i<values.length;i++) {
      html += `<option value=${values[i].PLACE_ID}>${values[i].PLACE_NAME}</option>`

    }

    return html;
  },activityCombobox:function(values) {
    var html = ''
    for (var i=0;i<values.length;i++) {
      html += `<option value=${values[i].ACTIVITY_ID}>${values[i].ACTIVITY_NAME}</option>`
    }

    return html;
  }
}
