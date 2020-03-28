#include "gtest/gtest.h"
#include "common_defs.h"
#include "str_util.h"
#include <string>
#include <ios>

namespace test_str_util {

    // The fixture for testing class Foo.

    class test_str_util : public ::testing::Test {
    protected:
        // You can remove any or all of the following functions if its body
        // is empty.

        test_str_util() {
            // You can do set-up work for each test here.
        }

        virtual ~test_str_util() {
            // You can do clean-up work that doesn't throw exceptions here.
        }

        // If the constructor and destructor are not enough for setting up
        // and cleaning up each test, you can define the following methods:

        virtual void SetUp() {
            // Code here will be called immediately after the constructor (right
            // before each test).
        }

        virtual void TearDown() {
            // Code here will be called immediately after each test (right
            // before the destructor).
        }

        // Objects declared here can be used by all tests in the test case for Foo.
    };

    // Tests that Foo does Xyz.

    TEST_F(test_str_util, ndays_to_string) {
        char buf[128];
        char *nilbuf {0};
        EXPECT_TRUE(ndays_to_string(-1.0, 0, buf));
        EXPECT_FALSE(ndays_to_string(0.0, -1, buf));
        EXPECT_STREQ(buf, "");
        EXPECT_TRUE(ndays_to_string(1.0, 0, nilbuf));
        EXPECT_FALSE(ndays_to_string(0.0, 0, buf));
        EXPECT_STREQ(buf, "0.00 sec ");
        EXPECT_FALSE(ndays_to_string(5, -1, buf));
        EXPECT_STREQ(buf, "5 days ");
        EXPECT_FALSE(ndays_to_string(1.234567890, 0, buf));
        EXPECT_STREQ(buf, "1 days 5 hr 37 min 46.67 sec ");
        EXPECT_FALSE(ndays_to_string(12.34567890, 1, buf));
        EXPECT_STREQ(buf, "12 days 8 hr 17.78 min ");
        EXPECT_FALSE(ndays_to_string(123.4567890, 2, buf));
        EXPECT_STREQ(buf, "123 days 10.96 hr ");
        EXPECT_FALSE(ndays_to_string(1234.567890, 3, buf));
        EXPECT_STREQ(buf, "3 yr 138.82 days ");
        EXPECT_FALSE(ndays_to_string(12345.67890, 4, buf));
        EXPECT_STREQ(buf, "33.801 yr ");
        EXPECT_FALSE(ndays_to_string(12345.67890, 5, buf));
        EXPECT_STREQ(buf, "");
        EXPECT_FALSE(ndays_to_string(1234.56789012345, 0, buf));
        EXPECT_STREQ(buf, "3 yr 138 days 13 hr 37 min 45.71 sec ");
    }

    TEST_F(test_str_util, secs_to_hmsf) {
        char buf[128];
        //char *nilbuf {0};
        secs_to_hmsf(0.0, buf);
        EXPECT_STREQ(buf, "0h00m00s00");
        secs_to_hmsf(1.0, buf);
        EXPECT_STREQ(buf, "0h00m01s00");
        secs_to_hmsf(1.23456789, buf);
        EXPECT_STREQ(buf, "0h00m01s23");
        secs_to_hmsf(12345.6789, buf);
        EXPECT_STREQ(buf, "3h25m45s67");
        secs_to_hmsf(123456.789, buf);
        EXPECT_STREQ(buf, "34h17m36s78");
    }

    TEST_F(test_str_util, nbytes_to_string) {
        char buf[256];
        nbytes_to_string(5.0, 0.0, buf, sizeof (buf));
        EXPECT_STREQ(buf, "5 bytes");
        nbytes_to_string(1024.0, 0.0, buf, sizeof (buf));
        EXPECT_STREQ(buf, "1.00 KB");
        nbytes_to_string(5.0*1024*1024, 0.0, buf, sizeof(buf));
        EXPECT_STREQ(buf, "5.00 MB");
        nbytes_to_string(15.0*1024*1024*1024, 0.0, buf, sizeof(buf));
        EXPECT_STREQ(buf, "15.00 GB");
        nbytes_to_string(50000000000000.0, 0.0, buf, sizeof (buf));
        EXPECT_STREQ(buf, "45.47 TB");
        nbytes_to_string(2.0, 48.0, buf, sizeof(buf));
        EXPECT_STREQ(buf, "2/48 bytes");
        nbytes_to_string(512.0, 1024.0, buf, sizeof (buf));
        EXPECT_STREQ(buf, "0.50/1.00 KB");
        nbytes_to_string(1024.0, 1.0 * 1024 * 1024, buf, sizeof (buf));
        EXPECT_STREQ(buf, "0.00/1.00 MB");
        nbytes_to_string(6.0*1024*1024*1024, 6.0*1024*1024*1024, buf, sizeof(buf));
        EXPECT_STREQ(buf, "6.00/6.00 GB");
        nbytes_to_string(24.0*1024*1024*1024*1024, 48.0*1024*1024*1024*1024, buf, sizeof(buf));
        EXPECT_STREQ(buf, "24.00/48.00 TB");
    }

    TEST_F(test_str_util, parse_command_line) {;
        char buf[256];
        char* argv[100];
        int ret;
        char *nilbuf {0};
        sprintf(buf, "one two three");
        ret = parse_command_line(buf, argv);
        EXPECT_EQ(ret, 3);
        EXPECT_STREQ(argv[0], "one");
        EXPECT_STREQ(argv[1], "two");
        EXPECT_STREQ(argv[2], "three");
        EXPECT_STREQ(argv[3], nilbuf);
        EXPECT_STREQ(buf, "one");
        sprintf(buf, "four \'five\' \"six\"");
        ret = parse_command_line(buf, argv);
        EXPECT_EQ(ret, 3);
        EXPECT_STREQ(argv[0], "four");
        EXPECT_STREQ(argv[1], "five");
        EXPECT_STREQ(argv[2], "six");
        EXPECT_STREQ(argv[3], nilbuf);
        EXPECT_STREQ(buf, "four");
        sprintf(buf, "seven \'eig ht\' \"ni ne\"");
        ret = parse_command_line(buf, argv);
        EXPECT_EQ(ret, 3);
        EXPECT_STREQ(argv[0], "seven");
        EXPECT_STREQ(argv[1], "eig ht");
        EXPECT_STREQ(argv[2], "ni ne");
        EXPECT_STREQ(argv[3], nilbuf);
        EXPECT_STREQ(buf, "seven");
        sprintf(buf, "tän \'elèv én\' \"tŵelv e\"");
        ret = parse_command_line(buf, argv);
        EXPECT_EQ(ret, 3);
        EXPECT_STREQ(argv[0], "tän");
        EXPECT_STREQ(argv[1], "elèv én");
        EXPECT_STREQ(argv[2], "tŵelv e");
        EXPECT_STREQ(argv[3], nilbuf);
        EXPECT_STREQ(buf, "tän");
        // function doesn't check syntax so this works too
        sprintf(buf, "13\" \'\"4teen\'\"   ");
        ret = parse_command_line(buf, argv);
        EXPECT_EQ(ret, 3);
        EXPECT_STREQ(argv[0], "13\"");
        EXPECT_STREQ(argv[1], "\"4teen");
        EXPECT_STREQ(argv[2], "   ");
        EXPECT_STREQ(argv[3], nilbuf);
        EXPECT_STREQ(buf, "13\"");
    }

    TEST_F(test_str_util, strip_whitespace) {
        std::string tmp = "     white space   ";
        strip_whitespace(tmp);
        EXPECT_EQ(tmp, "white space");
        tmp = "nospaces";
        strip_whitespace(tmp);
        EXPECT_EQ(tmp, "nospaces");
        char buf[128] = "     char space   ";
        strip_whitespace(buf);
        EXPECT_STREQ(buf, "char space");
    }

    TEST_F(test_str_util, strip_quotes) {
        std::string tmp = "\"\' white\' \"space\'\" ";
        strip_quotes(tmp);
        EXPECT_EQ(tmp, "white\' \"space");
        tmp = "no\"space\'s";
        strip_quotes(tmp);
        EXPECT_EQ(tmp, "no\"space\'s");
        tmp = "\"elèv\' én\"";
        strip_quotes(tmp);
        EXPECT_EQ(tmp, "elèv\' én");
        char buf[128] = "\"\' char\' \"space\'\" ";
        strip_quotes(buf);
        EXPECT_STREQ(buf, "char\' \"space");
    }

    TEST_F(test_str_util, collapse_whitespace) {
        std::string tmp = "     white space   ";
        collapse_whitespace(tmp);
        EXPECT_EQ(tmp, " white space ");
        tmp = "nospaces";
        collapse_whitespace(tmp);
        EXPECT_EQ(tmp, "nospaces");
        tmp = "inner     spaces";
        collapse_whitespace(tmp);
        EXPECT_EQ(tmp, "inner spaces");
        char buf[128] = "  char     spaces ";
        collapse_whitespace(buf);
        EXPECT_STREQ(buf, " char spaces ");
    }

    TEST_F(test_str_util, unescape_os_release) {
        //a-f get unescaped, g-j not
        char buf[128] = "a\\\\b\\$c\\\'d\\\"e\\`f\\\?g\\\th\\\bi\\12j";
        unescape_os_release(buf);
        EXPECT_STREQ(buf, "a\\b$c\'d\"e`f\\\?g\\\th\\\bi\\12j");
    }

    TEST_F(test_str_util, time_to_string) {
        char* buf;
        setenv("TZ", "UTC", 1);
        tzset();
        buf = time_to_string(false);
        EXPECT_STREQ(buf, "---");
        buf = time_to_string(1.0);
        EXPECT_STREQ(buf, "01-Jan-1970 00:00:01");
        buf = time_to_string(12345678910.0);
        EXPECT_STREQ(buf, "21-Mar-2361 19:15:10");
    }

    TEST_F(test_str_util, precision_time_to_string) {
        char* buf;
        setenv("TZ", "UTC", 1);
        tzset();
        buf = precision_time_to_string(false);
        EXPECT_STREQ(buf, "1970-01-01 00:00:00.0000");
        buf = precision_time_to_string(1.0);
        EXPECT_STREQ(buf, "1970-01-01 00:00:01.0000");
        buf = precision_time_to_string(1555876749.1234);
        EXPECT_STREQ(buf, "2019-04-21 19:59:09.1233");
        buf = precision_time_to_string(12345678910.10000);
        EXPECT_STREQ(buf, "2361-03-21 19:15:10.-2147483648");
    }

    TEST_F(test_str_util, timediff_format) {
        std::string tmp;
        tmp = timediff_format(false);
        EXPECT_EQ(tmp, "00:00:00");
        tmp = timediff_format(59.0);
        EXPECT_EQ(tmp, "00:00:59");
        tmp = timediff_format(3599.0);
        EXPECT_EQ(tmp, "00:59:59");
        tmp = timediff_format(3600.0);
        EXPECT_EQ(tmp, "01:00:00");
        tmp = timediff_format(123456.7);
        EXPECT_EQ(tmp, "1 days 10:17:36");
    }

    TEST_F(test_str_util, mysql_timestamp) {
        char buf[128];
        //char *nilbuf {0};
        mysql_timestamp(0.0, buf);
        EXPECT_STREQ(buf, "19700101000000");
        mysql_timestamp(1.0, buf);
        EXPECT_STREQ(buf, "19700101000001");
        mysql_timestamp(1555876749.1234, buf);
        EXPECT_STREQ(buf, "20190421195909");
        mysql_timestamp(12345678910.0, buf);
        EXPECT_STREQ(buf, "23610321191510");
    }

    TEST_F(test_str_util, string_substitute) {
        std::string tmp = "The quick brown fox jumps over the lazy dog";
        char buf[256];
        int ret = string_substitute(tmp.c_str(), buf, sizeof(buf), "brown", "red");
        EXPECT_EQ(ret, 0);
        EXPECT_STREQ(buf, "The quick red fox jumps over the lazy dog");
        ret = string_substitute(tmp.c_str(), buf, 13, "brown", "red");
        EXPECT_EQ(ret, ERR_BUFFER_OVERFLOW);
        ret = string_substitute(tmp.c_str(), buf, 16, "brown", "red");
        EXPECT_EQ(ret, ERR_BUFFER_OVERFLOW);
        tmp = "Falsches Üben von Xylophonmusik quält jeden größeren Zwerg";
        ret = string_substitute(tmp.c_str(), buf, sizeof(buf), "quält", "ärgert");
        EXPECT_EQ(ret, 0);
        EXPECT_STREQ(buf, "Falsches Üben von Xylophonmusik ärgert jeden größeren Zwerg");
    }

    TEST_F(test_str_util, strip_translation) {
        char buf[256] = "_(\"The quick brown fox jumps over the lazy dog\")";
        strip_translation(buf);
        EXPECT_STREQ(buf, "The quick brown fox jumps over the lazy dog");
        sprintf(buf, "The _(\"quick brown\") fox jumps over the _(\"lazy\") dog");
        strip_translation(buf);
        EXPECT_STREQ(buf, "The quick brown fox jumps over the lazy dog");
        sprintf(buf, "The _\"quick brown\" ) fox jumps over the (_\"lazy\") dog");
        strip_translation(buf);
        EXPECT_STREQ(buf, "The _\"quick brown\" ) fox jumps over the (_\"lazy dog");
    }

    TEST_F(test_str_util, lf_terminate) {
        char *buf;
        buf = (char*)malloc(256);
        strcpy(buf, "no\nlf ending");
        buf = lf_terminate(buf);
        EXPECT_STREQ(buf, "no\nlf ending\n");
        strcpy(buf, "lf\n ending\n");
        buf = lf_terminate(buf);
        EXPECT_STREQ(buf, "lf\n ending\n");
    }

    TEST_F(test_str_util, parse_serialnum) {
        char buf[256] = "[BOINC|1.2.3]", buf1[256], buf2[256], buf3[256];
        //sprintf(tmp, "[BOINC|1.2.3]");
        parse_serialnum(buf, buf1, buf2, buf3);
        EXPECT_STREQ(buf, "[BOINC|1.2.3]");
        EXPECT_STREQ(buf1, "[BOINC|1.2.3]");
        EXPECT_STREQ(buf2, "");
        EXPECT_STREQ(buf3, "");
        strcpy(buf, "[BOINC|1.2.3][vbox|4.5.6abc]");
        parse_serialnum(buf, buf1, buf2, buf3);
        EXPECT_STREQ(buf1, "[BOINC|1.2.3]");
        EXPECT_STREQ(buf2, "[vbox|4.5.6abc]");
        EXPECT_STREQ(buf3, "");
        strcpy(buf, "[BOINC|1.2.3][INTEL|Intel(R) HD Graphics|1|2406MB||201]");
        parse_serialnum(buf, buf1, buf2, buf3);
        EXPECT_STREQ(buf1, "[BOINC|1.2.3]");
        EXPECT_STREQ(buf2, "");
        EXPECT_STREQ(buf3, "[INTEL|Intel(R) HD Graphics|1|2406MB||201]");
        strcpy(buf, "[vbox|4.5.6abc][INTEL|Intel(R) HD Graphics|1|2406MB||201][BOINC|1.2.3]");
        parse_serialnum(buf, buf1, buf2, buf3);
        EXPECT_STREQ(buf1, "[BOINC|1.2.3]");
        EXPECT_STREQ(buf2, "[vbox|4.5.6abc]");
        EXPECT_STREQ(buf3, "[INTEL|Intel(R) HD Graphics|1|2406MB||201]");
        strcpy(buf, "[BOINC|1.2.3][INTEL|Intel(R) HD Graphics|1|2406MB||201][vbox|4.5.6abc]");
        parse_serialnum(buf, buf1, buf2, buf3);
        EXPECT_STREQ(buf1, "[BOINC|1.2.3]");
        EXPECT_STREQ(buf2, "[vbox|4.5.6abc]");
        EXPECT_STREQ(buf3, "[INTEL|Intel(R) HD Graphics|1|2406MB||201]");
        strcpy(buf, "[BOINC|1.2.3][vbox|4.5.6abc][INTEL|Intel(R) HD Graphics|1|2406MB||201]");
        parse_serialnum(buf, buf1, buf2, buf3);
        EXPECT_STREQ(buf1, "[BOINC|1.2.3]");
        EXPECT_STREQ(buf2, "[vbox|4.5.6abc]");
        EXPECT_STREQ(buf3, "[INTEL|Intel(R) HD Graphics|1|2406MB||201]");
        strcpy(buf, "[BOINC|7.6.22][CAL|ATI Radeon HD 5800/5900 series (Cypress/Hemlock)|2|1024MB|1.4.1848|102][vbox|5.1.26]");
        parse_serialnum(buf, buf1, buf2, buf3);
        EXPECT_STREQ(buf1, "[BOINC|7.6.22]");
        EXPECT_STREQ(buf2, "[vbox|5.1.26]");
        EXPECT_STREQ(buf3, "[CAL|ATI Radeon HD 5800/5900 series (Cypress/Hemlock)|2|1024MB|1.4.1848|102]");
        strcpy(buf, "[BOINC|7.6.22[CAL|ATI Radeon HD 5800/5900 series (Cypress/Hemlock)|2|1024MB|1.4.1848|102][vbox|5.1.26]");
        parse_serialnum(buf, buf1, buf2, buf3);
        EXPECT_STREQ(buf1, "[BOINC|7.6.22[CAL|ATI Radeon HD 5800/5900 series (Cypress/Hemlock)|2|1024MB|1.4.1848|102]");
        EXPECT_STREQ(buf2, "[vbox|5.1.26]");
        EXPECT_STREQ(buf3, "");
        strcpy(buf, "[BOINC|7.6.22][CAL|ATI Radeon HD 5800/5900 series [Cypress/Hemlock]|2|1024MB|1.4.1848|102][vbox|5.1.26]");
        parse_serialnum(buf, buf1, buf2, buf3);
        EXPECT_STREQ(buf1, "[BOINC|7.6.22]");
        EXPECT_STREQ(buf2, "");
        EXPECT_STREQ(buf3, "[CAL|ATI Radeon HD 5800/5900 series [Cypress/Hemlock]");
        strcpy(buf, "[BOINC|7.6.22][CAL|ATI Radeon HD 5800/5900 series (Cypress/Hemlock)|2|1024MB|1.4.1848|102][extra|7.8.9][vbox|5.1.26]");
        parse_serialnum(buf, buf1, buf2, buf3);
        EXPECT_STREQ(buf1, "[BOINC|7.6.22]");
        EXPECT_STREQ(buf2, "[vbox|5.1.26]");
        //TODO: fix parse_serialnum so this doesn't happen:
        EXPECT_STREQ(buf3, "[CAL|ATI Radeon HD 5800/5900 series (Cypress/Hemlock)|2|1024MB|1.4.1848|102][extra|7.8.9]");
    }

    TEST_F(test_str_util, is_valid_filename) {
        //char tmp = "filename.txt";
        bool ret = is_valid_filename("filename.txt");
        EXPECT_TRUE(ret);
        ret = is_valid_filename("../filename.txt");
        EXPECT_FALSE(ret);
        ret = is_valid_filename("../file\nname.txt");
        EXPECT_FALSE(ret);
        ret = is_valid_filename("/filename.txt");
        EXPECT_FALSE(ret);
    }

    TEST_F(test_str_util, path_to_filename) {
        std::string fname = "";
        EXPECT_EQ(path_to_filename("/home/blah", fname), 0);
        EXPECT_EQ(fname, "blah");
        EXPECT_EQ(path_to_filename("hellokeith", fname), 0);
        EXPECT_EQ(fname, "hellokeith");
        fname = "";
        EXPECT_EQ(path_to_filename("/home/blah/", fname), -2);
        EXPECT_EQ(fname, "");
        EXPECT_EQ(path_to_filename("", fname), -1);
        EXPECT_EQ(fname, "");
        char *buf;
        EXPECT_EQ(path_to_filename("/home/blah", buf), 0);
        EXPECT_STREQ(buf, "blah");
        EXPECT_EQ(path_to_filename("hellokeith", buf), 0);
        EXPECT_STREQ(buf, "hellokeith");
        strcpy(buf, "");
        EXPECT_EQ(path_to_filename("/home/blah/", buf), -2);
        EXPECT_STREQ(buf, "");
        EXPECT_EQ(path_to_filename("", buf), -1);
        EXPECT_STREQ(buf, "");
    }

} // namespace
