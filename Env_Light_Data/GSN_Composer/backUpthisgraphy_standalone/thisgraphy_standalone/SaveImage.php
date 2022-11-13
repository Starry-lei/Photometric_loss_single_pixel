<?php

$fileName = filter_input(INPUT_POST, 'fileName', FILTER_VALIDATE_REGEXP, array("options" => array('regexp' => '/^[A-Za-z0-9]*$/')));
$format = filter_input(INPUT_POST, 'format', FILTER_VALIDATE_INT);
$quality = filter_input(INPUT_POST, 'quality', FILTER_VALIDATE_INT);
$imgData = file_get_contents($_FILES['imgData']['tmp_name']);

if(empty($fileName)) {
  die( 'The file name is invalid.');
}

$file = "./image_".$fileName.".png";
$complete = 0;

if ($format == 0 || $format == 1) { // format is png or jpg
  $cleaned = str_replace(' ', '+', $imgData);
  // remove data:image/png;base64
  $removedBase64Header = substr($cleaned, strpos($cleaned, ",") + 1);

  $raw = base64_decode($removedBase64Header);
  if ($raw == false) {
    die('base64_decode returns false.');
  }
  $complete = $raw;
}

if ($format == 1) { // format is jpg
  $file = "./image_".$fileName."_pngtojpg.png";
}

if ($format == 2) { // format is pfm
  $file = "./image_".$fileName.".pfm";
  $complete = $imgData;
}

if ($format == 3) { // format is hdr
  $file = "./image_".$fileName.".hdr";
  $complete = $imgData;
}

// write file
file_put_contents($file, $complete);

if ($format == 1) { // convert png to jpg
  $fileJpg = "./image_".$fileName.".jpg";
  $image = imagecreatefrompng($file);
  $bg = imagecreatetruecolor(imagesx($image), imagesy($image));
  imagefill($bg, 0, 0, imagecolorallocate($bg, 255, 255, 255));
  imagealphablending($bg, TRUE);
  imagecopy($bg, $image, 0, 0, 0, 0, imagesx($image), imagesy($image));
  imagedestroy($image);
  imagejpeg($bg, $fileJpg, $quality);
  imagedestroy($bg);
  unlink($file); // delete png file
}

echo "Success";

?>