$(function () {
  $("#loader").hide();
  $("#time").hide();
  $("#en_2_in_form").on("submit", function (e) {
    e.preventDefault(); //prevent form from submitting
    $("#loader").show();
    $("#time").show();
    $("#output").text("");
    var url = "/projects/en_2_in";
    $.ajax({
      type: "POST",
      url: url,
      data: {
        english: $("#english").val(),
        csrfmiddlewaretoken: $("input[name=csrfmiddlewaretoken]").val(),
        dataType: "json",
      },

      success: function (response) {
        $("#loader").hide();
        $("#time").hide();
        var out = response.output;
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
