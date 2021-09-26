$(function () {
  $("#loader").hide();
  $("#time").hide();
  $("#traffic_sign_form").on("submit", function (e) {
    e.preventDefault(); //prevent form from submitting
    $("#loader").show();
    $("#time").show();
    $("#output").text("");
    var f_obj = $("#file_upload").get(0).files[0];

    var data = new FormData();
    var csrftoken = $("input[name=csrfmiddlewaretoken]").val();
    data.append("file", f_obj);
    var url = "/projects/traffic";
    $.ajax({
      type: "POST",
      headers: { "X-CSRFToken": csrftoken },
      url: url,
      data: data,
      cache: false,
      processData: false,
      contentType: false,

      success: function (response) {
        $("#loader").hide();
        $("#time").hide();
        var out = "PREDICTED SIGN: " + response.sign;
        // console.log(out);
        $("#output").text(out);
      },

      failure: function () {
        $("#loader").hide();
        $("#time").hide();
        $("#output").text("<ERROR>Something Went Wrong.<ERROR>");
        console.log("Errored");
      },
    });
  });
});
