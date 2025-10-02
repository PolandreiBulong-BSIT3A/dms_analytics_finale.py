-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Oct 02, 2025 at 04:01 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `ispsc_tagudin_dms_2`
--

-- --------------------------------------------------------

--
-- Table structure for table `action_required`
--

CREATE TABLE `action_required` (
  `action_id` int(11) NOT NULL,
  `action_name` varchar(255) NOT NULL,
  `action_description` text DEFAULT NULL,
  `action_category` enum('decision','communication','document_management','administrative','custom') DEFAULT 'custom',
  `is_active` tinyint(1) DEFAULT 1,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `created_by` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `action_required`
--

INSERT INTO `action_required` (`action_id`, `action_name`, `action_description`, `action_category`, `is_active`, `created_at`, `updated_at`, `created_by`) VALUES
(1, 'Appropriate Action', 'Take appropriate action based on document content', 'decision', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(2, 'Study/Information', 'Study the document for information purposes', 'decision', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(3, 'Signature', 'Document requires signature', 'decision', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(4, 'Clearance/Approval', 'Document requires clearance or approval', 'decision', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(5, 'Comments/Feedback', 'Provide comments or feedback on the document', 'communication', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(6, 'Reply', 'Reply to the document or sender', 'communication', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(7, 'Take up with me', 'Discuss the document in person', 'communication', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(8, 'File', 'File the document for record keeping', 'document_management', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(9, 'Return', 'Return the document to sender', 'document_management', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(10, 'Submit Report', 'Submit a report based on the document', 'document_management', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(11, 'Submit CSW', 'Submit CSW (Complete Staff Work)', 'document_management', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(12, 'Attendance', 'Mark attendance or participation', 'administrative', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(13, 'Review', 'Review the document thoroughly', 'decision', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(14, 'Prepare Memo/Special', 'Prepare a memo or special order', 'document_management', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(15, 'Order/Office Order', 'Issue an order or office order', 'document_management', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1),
(16, 'Representation', 'Handle representation on behalf of', 'administrative', 1, '2024-12-31 16:00:00', '2024-12-31 16:00:00', 1);

-- --------------------------------------------------------

--
-- Table structure for table `announcements`
--

CREATE TABLE `announcements` (
  `announcement_id` int(11) NOT NULL,
  `title` varchar(255) NOT NULL,
  `body` text NOT NULL,
  `visible_to_all` tinyint(1) DEFAULT 1,
  `status` enum('draft','scheduled','published','archived') DEFAULT 'draft',
  `publish_at` datetime DEFAULT NULL,
  `expire_at` datetime DEFAULT NULL,
  `created_by_name` varchar(150) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `announcements`
--

INSERT INTO `announcements` (`announcement_id`, `title`, `body`, `visible_to_all`, `status`, `publish_at`, `expire_at`, `created_by_name`, `created_at`, `updated_at`) VALUES
(11, 'ALL MEMBERS', 'Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.\n\nLorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.\n\nLorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.\n\nLorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.\n\nLorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.', 1, 'published', '2025-09-07 09:15:50', NULL, 'THIS IS ME', '2025-09-07 09:15:50', '2025-09-30 21:30:34'),
(12, 'ALL', 'ADS fsfasd', 1, 'published', '2025-09-07 10:09:55', NULL, 'THIS IS ME', '2025-09-07 10:09:55', '2025-09-30 21:30:34'),
(22, 'DEAN CASS', 'Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.\n\nLorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.\n\nLorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.\n\nLorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.\n\nLorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.', 0, 'published', '2025-09-27 09:34:02', NULL, 'THIS IS ME', '2025-09-27 01:34:02', '2025-09-30 21:30:34'),
(25, 'ALL', 'a', 1, 'published', '2025-10-01 05:07:04', NULL, 'THIS IS ME', '2025-09-30 21:07:04', '2025-09-30 21:30:34'),
(26, 'sadasdasd', 'sdasd', 1, 'published', '2025-10-01 05:34:47', NULL, 'THIS IS ME', '2025-09-30 21:34:47', '2025-09-30 21:34:47'),
(27, 'dasdasdasdasdsa', 'sda', 0, 'published', '2025-10-01 05:38:32', NULL, 'Levv1', '2025-09-30 21:38:32', '2025-09-30 21:38:32'),
(28, 'asd', 'asd', 1, 'published', '2025-10-01 05:40:06', NULL, 'THIS IS ME', '2025-09-30 21:40:06', '2025-09-30 21:40:06'),
(29, 'TEST 1 FOR ALL', '123312', 0, 'published', '2025-10-01 05:40:43', NULL, 'Victoriya Visha Von Klauss', '2025-09-30 21:40:43', '2025-09-30 21:40:43'),
(30, 'swer', 'ewr', 0, 'published', '2025-10-01 07:14:38', NULL, 'THIS IS ME', '2025-09-30 23:14:38', '2025-09-30 23:14:38');

-- --------------------------------------------------------

--
-- Table structure for table `announcement_departments`
--

CREATE TABLE `announcement_departments` (
  `announcement_id` int(11) NOT NULL,
  `department_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `announcement_departments`
--

INSERT INTO `announcement_departments` (`announcement_id`, `department_id`) VALUES
(22, 2),
(27, 2),
(29, 2),
(30, 1);

-- --------------------------------------------------------

--
-- Table structure for table `announcement_roles`
--

CREATE TABLE `announcement_roles` (
  `announcement_id` int(11) NOT NULL,
  `role` enum('ADMIN','DEAN','FACULTY') NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `announcement_roles`
--

INSERT INTO `announcement_roles` (`announcement_id`, `role`) VALUES
(29, 'ADMIN'),
(29, 'DEAN'),
(29, 'FACULTY');

-- --------------------------------------------------------

--
-- Table structure for table `announcement_users`
--

CREATE TABLE `announcement_users` (
  `announcement_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `departments`
--

CREATE TABLE `departments` (
  `department_id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `code` varchar(20) DEFAULT NULL,
  `is_active` tinyint(1) DEFAULT 1,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `departments`
--

INSERT INTO `departments` (`department_id`, `name`, `code`, `is_active`, `created_at`, `updated_at`) VALUES
(1, 'College of Arts and Sciences', 'CAS', 1, '2025-09-30 01:29:43', '2025-09-30 01:29:43'),
(2, 'College of Management and Business Economics', 'CMBE', 1, '2025-09-30 01:29:43', '2025-09-30 01:29:43'),
(3, 'College of Teacher Education', 'CTE', 1, '2025-09-30 01:29:43', '2025-09-30 01:29:43'),
(4, 'Laboratory High School', 'LHS', 1, '2025-09-30 01:29:43', '2025-09-30 01:29:43'),
(6, 'Graduate School', 'GRADUATE SCHOOL', 1, '2025-09-30 01:29:43', '2025-09-30 01:29:43'),
(999, 'Unassigned', 'UNASSIGNED', 1, '2025-09-30 02:34:44', '2025-09-30 02:34:44');

-- --------------------------------------------------------

--
-- Table structure for table `dms_documents`
--

CREATE TABLE `dms_documents` (
  `doc_id` int(11) NOT NULL,
  `doc_type` int(11) NOT NULL,
  `folder_id` int(11) DEFAULT NULL,
  `reference` varchar(255) DEFAULT NULL,
  `title` varchar(255) NOT NULL,
  `revision` varchar(50) DEFAULT NULL,
  `rev_date` date DEFAULT NULL,
  `from_field` varchar(255) DEFAULT NULL,
  `to_field` varchar(255) DEFAULT NULL,
  `date_received` date DEFAULT NULL,
  `google_drive_link` text DEFAULT NULL,
  `description` text DEFAULT NULL,
  `available_copy` enum('hard_copy','soft_copy','both') DEFAULT 'soft_copy',
  `received_by` varchar(255) DEFAULT NULL,
  `received_by_user_id` int(11) DEFAULT NULL,
  `visible_to_all` tinyint(1) DEFAULT 1,
  `allowed_user_ids` text DEFAULT NULL,
  `allowed_roles` text DEFAULT NULL,
  `status` enum('active','inactive','pending','deleted') DEFAULT 'active',
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `created_by_name` varchar(150) DEFAULT NULL,
  `deleted` tinyint(1) DEFAULT 0,
  `deleted_at` datetime DEFAULT NULL,
  `deleted_by_name` varchar(150) DEFAULT NULL,
  `restored_at` datetime DEFAULT NULL,
  `restored_by_name` varchar(150) DEFAULT NULL,
  `is_reply_to_doc_id` int(11) DEFAULT NULL,
  `reply_type` varchar(50) DEFAULT NULL,
  `created_by_user_id` int(11) DEFAULT NULL,
  `visibility` enum('ALL','DEPARTMENT','SPECIFIC_USERS','SPECIFIC_ROLES','ROLE_DEPARTMENT') DEFAULT 'ALL',
  `target_users` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`target_users`)),
  `target_roles` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`target_roles`)),
  `target_role_dept` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`target_role_dept`))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `dms_documents`
--

INSERT INTO `dms_documents` (`doc_id`, `doc_type`, `folder_id`, `reference`, `title`, `revision`, `rev_date`, `from_field`, `to_field`, `date_received`, `google_drive_link`, `description`, `available_copy`, `received_by`, `received_by_user_id`, `visible_to_all`, `allowed_user_ids`, `allowed_roles`, `status`, `created_at`, `updated_at`, `created_by_name`, `deleted`, `deleted_at`, `deleted_by_name`, `restored_at`, `restored_by_name`, `is_reply_to_doc_id`, `reply_type`, `created_by_user_id`, `visibility`, `target_users`, `target_roles`, `target_role_dept`) VALUES
(99, 5, 7, 'dsds', 'Sdds', NULL, NULL, 'Admin', 'Victoria', '2025-10-01', 'https://drive.google.com/file/d/1kMbzNo_DSXpp4HFIXX8FCsUUixVEsReG/view?usp=drive_link', NULL, 'soft_copy', NULL, NULL, 1, NULL, NULL, 'active', '2025-09-30 20:32:21', '2025-09-30 21:30:34', 'THIS IS ME', 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'ALL', NULL, NULL, NULL),
(100, 5, 5, 'asd', 'Asd', NULL, NULL, 'Sad', 'Victoria', '2025-10-01', 'https://drive.google.com/file/d/1kMbzNo_DSXpp4HFIXX8FCsUUixVEsReG/view?usp=drive_link', NULL, 'soft_copy', NULL, NULL, 0, '11', NULL, 'active', '2025-09-30 21:13:45', '2025-09-30 21:30:34', 'THIS IS ME', 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'ALL', NULL, NULL, NULL),
(101, 5, 8, 'ASDASD', 'Mobile Test', NULL, NULL, 'Admin', 'Victoria', '2025-10-01', 'https://drive.google.com/file/d/1TAVFyRUbKhjHtrqaCL_ASPVMovigMbur/view?usp=drive_link', NULL, 'both', NULL, NULL, 1, NULL, NULL, 'active', '2025-10-01 08:38:48', '2025-10-01 08:38:48', 'THIS IS ME', 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'ALL', NULL, NULL, NULL),
(102, 5, NULL, 'REPLY-1759313597011', 'asdas', NULL, NULL, 'ADMIN LAN LAN', 'System', '2025-10-01', 'https://drive.google.com/file/d/1sLmzWc-viPP9l5sRvH6tG5NS7iSFJ1YS/view?usp=drive_link', '', 'soft_copy', NULL, NULL, 1, NULL, NULL, 'active', '2025-10-01 10:13:17', '2025-10-01 10:13:17', 'ADMIN LAN LAN', 0, NULL, NULL, NULL, NULL, 100, 'action_response', 1, 'ALL', NULL, NULL, NULL);

-- --------------------------------------------------------

--
-- Table structure for table `dms_user`
--

CREATE TABLE `dms_user` (
  `user_id` int(11) NOT NULL,
  `profile_pic` longtext DEFAULT NULL,
  `user_email` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  `Username` varchar(100) NOT NULL,
  `firstname` varchar(100) NOT NULL,
  `lastname` varchar(100) NOT NULL,
  `Contact_number` varchar(20) DEFAULT NULL,
  `department_id` int(11) DEFAULT NULL,
  `role` enum('ADMIN','DEAN','FACULTY') DEFAULT 'FACULTY',
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `status` enum('active','inactive','banned','pending','deleted') DEFAULT 'pending',
  `is_verified` enum('yes','no') DEFAULT 'no',
  `verification_token` varchar(255) DEFAULT NULL,
  `verification_code` varchar(10) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `dms_user`
--

INSERT INTO `dms_user` (`user_id`, `profile_pic`, `user_email`, `password`, `Username`, `firstname`, `lastname`, `Contact_number`, `department_id`, `role`, `created_at`, `updated_at`, `status`, `is_verified`, `verification_token`, `verification_code`) VALUES
(1, 'https://lh3.googleusercontent.com/a/ACg8ocIhpD_184y_Edc-9RJod2dRMKXfsKksA7Z1AKaWM-5lc7srqUFH=s96-c', 'polandreiladera03@gmail.com', '', 'THIS IS ME', 'ADMIN LAN', 'LAN', '09184226085', 1, 'ADMIN', '2025-09-07 03:34:43', '2025-09-30 21:30:34', 'active', 'yes', NULL, NULL),
(3, 'https://lh3.googleusercontent.com/a/ACg8ocLCtLZz7Zmt471wn_Ox2BYg5YUHStURu5kczNi_W3goIXHxvHw=s96-c', 'laderasanjuan03@gmail.com', '', 'Levv1', 'asd', 'asd', '1112121212', 2, 'DEAN', '2025-09-07 03:42:46', '2025-09-23 01:15:02', 'active', 'yes', NULL, NULL),
(9, 'https://lh3.googleusercontent.com/a/ACg8ocKY7FHsiOFJd8h-754VQgb1muTsSXKonCDewIZtAZ5GR1itVQ=s96-c', 'capstoneuser101@gmail.com', '', 'capstone 101', 'casp', '1', '09184226085', 1, 'FACULTY', '2025-09-14 22:22:57', '2025-09-14 22:27:26', 'active', 'yes', NULL, NULL),
(11, 'https://lh3.googleusercontent.com/a/ACg8ocJgAiBETzRmatXaHQCEnhQAHqiISli8O_R1o7WzyaOC6LJFmXtg=s96-c', 'polsanjuanladera03@gmail.com', '', 'Pol Andrei L. Bulong', 'ad', 'pol', '09184226085', 2, 'FACULTY', '2025-09-22 03:30:36', '2025-09-30 02:05:33', 'active', 'yes', NULL, NULL),
(25, 'https://example.com/hsr/avatar01.png', 'hsr_trailblazer01@hsr.test', '', 'hsr_trailblazer01', 'Himeko', 'Astral', '09280000001', 1, 'FACULTY', '2025-09-22 02:00:01', '2025-09-22 02:00:01', 'active', 'yes', NULL, NULL),
(26, 'https://example.com/hsr/avatar02.png', 'hsr_trailblazer02@hsr.test', '', 'hsr_trailblazer02', 'Welt', 'Yang', '09280000002', 2, 'FACULTY', '2025-09-22 02:01:02', '2025-09-22 02:01:02', 'active', 'yes', NULL, NULL),
(27, 'https://example.com/hsr/avatar03.png', 'hsr_trailblazer03@hsr.test', '', 'hsr_trailblazer03', 'March', 'Seven', '09280000003', 3, 'FACULTY', '2025-09-22 02:02:03', '2025-09-22 02:02:03', 'active', 'yes', NULL, NULL),
(28, 'https://example.com/hsr/avatar04.png', 'hsr_trailblazer04@hsr.test', '', 'hsr_trailblazer04', 'Dan', 'Heng', '09280000004', 4, 'FACULTY', '2025-09-22 02:03:04', '2025-09-22 02:03:04', 'active', 'yes', NULL, NULL),
(30, 'https://example.com/hsr/avatar06.png', 'hsr_trailblazer06@hsr.test', '', 'hsr_trailblazer06', 'Seele', 'Vollerei', '09280000006', 1, 'FACULTY', '2025-09-22 02:05:06', '2025-09-22 02:05:06', 'active', 'yes', NULL, NULL),
(31, 'https://example.com/hsr/avatar07.png', 'hsr_trailblazer07@hsr.test', '', 'hsr_trailblazer07', 'Kafka', 'Stellar', '09280000007', 2, 'FACULTY', '2025-09-22 02:06:07', '2025-09-22 02:06:07', 'active', 'yes', NULL, NULL),
(32, 'https://example.com/hsr/avatar08.png', 'hsr_trailblazer08@hsr.test', '', 'hsr_trailblazer08', 'Silver', 'Wolf', '09280000008', 3, 'FACULTY', '2025-09-22 02:07:08', '2025-09-22 02:07:08', 'active', 'yes', NULL, NULL),
(33, 'https://example.com/hsr/avatar09.png', 'hsr_trailblazer09@hsr.test', '', 'hsr_trailblazer09', 'Blade', 'Ebon', '09280000009', 4, 'FACULTY', '2025-09-22 02:08:09', '2025-09-22 02:08:09', 'active', 'yes', NULL, NULL),
(34, 'https://example.com/hsr/avatar10.png', 'hsr_trailblazer10@hsr.test', '', 'hsr_trailblazer10', 'Luocha', 'Seraph', '09280000010', NULL, 'FACULTY', '2025-09-22 02:09:10', '2025-09-30 02:03:15', 'active', 'yes', NULL, NULL),
(35, 'https://example.com/hsr/avatar11.png', 'hsr_trailblazer11@hsr.test', '', 'hsr_trailblazer11', 'Clara', 'Guardian', '09280000011', 1, 'FACULTY', '2025-09-22 02:10:11', '2025-09-22 02:10:11', 'active', 'yes', NULL, NULL),
(36, 'https://example.com/hsr/avatar12.png', 'hsr_trailblazer12@hsr.test', '', 'hsr_trailblazer12', 'Sampo', 'Koski', '09280000012', 2, 'FACULTY', '2025-09-22 02:11:12', '2025-09-22 02:11:12', 'active', 'yes', NULL, NULL),
(37, 'https://example.com/hsr/avatar13.png', 'hsr_trailblazer13@hsr.test', '', 'hsr_trailblazer13', 'Tingyun', 'Aurora', '09280000013', 3, 'FACULTY', '2025-09-22 02:12:13', '2025-09-22 02:12:13', 'active', 'yes', NULL, NULL),
(38, 'https://example.com/hsr/avatar14.png', 'hsr_trailblazer14@hsr.test', '', 'hsr_trailblazer14', 'Jing', 'Yuan', '09280000014', 4, 'FACULTY', '2025-09-22 02:13:14', '2025-09-22 02:13:14', 'active', 'yes', NULL, NULL),
(39, 'https://example.com/hsr/avatar15.png', 'hsr_trailblazer15@hsr.test', '', 'hsr_trailblazer15', 'Yanqing', 'General', '09280000015', NULL, 'FACULTY', '2025-09-22 02:14:15', '2025-09-30 02:03:15', 'active', 'yes', NULL, NULL),
(41, 'https://example.com/hsr/avatar17.png', 'hsr_trailblazer17@hsr.test', '', 'hsr_trailblazer17', 'Qingque', 'Mahjong', '09280000017', 2, 'FACULTY', '2025-09-22 02:16:17', '2025-09-22 02:16:17', 'active', 'yes', NULL, NULL),
(42, 'https://example.com/hsr/avatar18.png', 'hsr_trailblazer18@hsr.test', '', 'hsr_trailblazer18', 'Herta', 'Genius', '09280000018', 3, 'FACULTY', '2025-09-22 02:17:18', '2025-09-22 02:17:18', 'active', 'yes', NULL, NULL),
(43, 'https://example.com/hsr/avatar19.png', 'hsr_trailblazer19@hsr.test', '', 'hsr_trailblazer19', 'Pela', 'Warrior', '09280000019', 4, 'FACULTY', '2025-09-22 02:18:19', '2025-09-22 02:18:19', 'active', 'yes', NULL, NULL),
(44, 'https://example.com/hsr/avatar20.png', 'hsr_trailblazer20@hsr.test', '', 'hsr_trailblazer20', 'Lynx', 'Support', '09280000020', NULL, 'FACULTY', '2025-09-22 02:19:20', '2025-09-30 02:03:15', 'active', 'yes', NULL, NULL),
(45, 'https://example.com/hsr/avatar21.png', 'hsr_trailblazer21@hsr.test', '', 'hsr_trailblazer21', 'Topaz', 'Interastral', '09280000021', 1, 'FACULTY', '2025-09-22 03:00:21', '2025-09-22 03:00:21', 'active', 'yes', NULL, NULL),
(46, 'https://example.com/hsr/avatar22.png', 'hsr_trailblazer22@hsr.test', '', 'hsr_trailblazer22', 'Numby', 'Trotter', '09280000022', 2, 'FACULTY', '2025-09-22 03:01:22', '2025-09-22 03:01:22', 'active', 'yes', NULL, NULL),
(47, 'https://example.com/hsr/avatar23.png', 'hsr_trailblazer23@hsr.test', '', 'hsr_trailblazer23', 'Guinaifen', 'Performer', '09280000023', 3, 'FACULTY', '2025-09-22 03:02:23', '2025-09-22 03:02:23', 'active', 'yes', NULL, NULL),
(48, 'https://example.com/hsr/avatar24.png', 'hsr_trailblazer24@hsr.test', '', 'hsr_trailblazer24', 'Fu', 'Xuan', '09280000024', 4, 'FACULTY', '2025-09-22 03:03:24', '2025-09-22 03:03:24', 'active', 'yes', NULL, NULL),
(49, 'https://example.com/hsr/avatar25.png', 'hsr_trailblazer25@hsr.test', '', 'hsr_trailblazer25', 'Jingliu', 'Swordmaster', '09280000025', NULL, 'FACULTY', '2025-09-22 03:04:25', '2025-09-30 02:03:15', 'active', 'yes', NULL, NULL),
(50, 'https://example.com/hsr/avatar26.png', 'hsr_trailblazer26@hsr.test', '', 'hsr_trailblazer26', 'Xueyi', 'Judge', '09280000026', 1, 'FACULTY', '2025-09-22 03:05:26', '2025-09-22 03:05:26', 'active', 'yes', NULL, NULL),
(51, 'https://example.com/hsr/avatar27.png', 'hsr_trailblazer27@hsr.test', '', 'hsr_trailblazer27', 'Dr.', 'Ratio', '09280000027', 2, 'FACULTY', '2025-09-22 03:06:27', '2025-09-22 03:06:27', 'active', 'yes', NULL, NULL),
(52, 'https://example.com/hsr/avatar28.png', 'hsr_trailblazer28@hsr.test', '', 'hsr_trailblazer28', 'Black', 'Swan', '09280000028', 3, 'FACULTY', '2025-09-22 03:07:28', '2025-09-22 03:07:28', 'active', 'yes', NULL, NULL),
(53, 'https://example.com/hsr/avatar29.png', 'hsr_trailblazer29@hsr.test', '', 'hsr_trailblazer29', 'Sparkle', 'Mask', '09280000029', 4, 'FACULTY', '2025-09-22 03:08:29', '2025-09-22 03:08:29', 'active', 'yes', NULL, NULL),
(55, 'https://example.com/hsr/avatar31.png', 'hsr_trailblazer31@hsr.test', '', 'hsr_trailblazer31', 'Robin', 'Songstress', '09280000031', 1, 'FACULTY', '2025-09-22 03:10:31', '2025-09-22 03:10:31', 'active', 'yes', NULL, NULL),
(56, 'https://example.com/hsr/avatar32.png', 'hsr_trailblazer32@hsr.test', '', 'hsr_trailblazer32', 'Firefly', 'Stellar', '09280000032', 2, 'FACULTY', '2025-09-22 03:11:32', '2025-09-22 03:11:32', 'active', 'yes', NULL, NULL),
(57, 'https://example.com/hsr/avatar33.png', 'hsr_trailblazer33@hsr.test', '', 'hsr_trailblazer33', 'Ruan', 'Mei', '09280000033', 3, 'FACULTY', '2025-09-22 03:12:33', '2025-09-22 03:12:33', 'active', 'yes', NULL, NULL),
(58, 'https://example.com/hsr/avatar34.png', 'hsr_trailblazer34@hsr.test', '', 'hsr_trailblazer34', 'Gallagher', 'Medic', '09280000034', 4, 'FACULTY', '2025-09-22 03:13:34', '2025-09-22 03:13:34', 'active', 'yes', NULL, NULL),
(60, 'https://example.com/hsr/avatar36.png', 'hsr_trailblazer36@hsr.test', '', 'hsr_trailblazer36', 'Jade', 'Ten', '09280000036', 1, 'FACULTY', '2025-09-22 03:15:36', '2025-09-22 03:15:36', 'active', 'yes', NULL, NULL),
(61, 'https://example.com/hsr/avatar37.png', 'hsr_trailblazer37@hsr.test', '', 'hsr_trailblazer37', 'Misha', 'Belobog', '09280000037', 2, 'FACULTY', '2025-09-22 03:16:37', '2025-09-22 03:16:37', 'active', 'yes', NULL, NULL),
(62, 'https://example.com/hsr/avatar38.png', 'hsr_trailblazer38@hsr.test', '', 'hsr_trailblazer38', 'Sunday', 'Harmony', '09280000038', 3, 'FACULTY', '2025-09-22 03:17:38', '2025-09-22 03:17:38', 'active', 'yes', NULL, NULL),
(64, 'https://example.com/hsr/avatar40.png', 'hsr_trailblazer40@hsr.test', '', 'hsr_trailblazer40', 'Sam', 'Mecha', '09280000040', NULL, 'FACULTY', '2025-09-22 03:19:40', '2025-09-30 02:03:15', 'active', 'yes', NULL, NULL),
(65, 'https://example.com/hsr/avatar41.png', 'hsr_trailblazer41@hsr.test', '', 'hsr_trailblazer41', 'Jing', 'Yuan', '09280000041', 1, 'FACULTY', '2025-09-22 03:20:41', '2025-09-22 03:20:41', 'active', 'yes', NULL, NULL),
(66, 'https://example.com/hsr/avatar42.png', 'hsr_trailblazer42@hsr.test', '', 'hsr_trailblazer42', 'Yanqing', 'Prodigy', '09280000042', 2, 'FACULTY', '2025-09-22 03:21:42', '2025-09-22 03:21:42', 'active', 'yes', NULL, NULL),
(67, 'https://example.com/hsr/avatar43.png', 'hsr_trailblazer43@hsr.test', '', 'hsr_trailblazer43', 'Dan', 'Heng', '09280000043', 3, 'FACULTY', '2025-09-22 03:22:43', '2025-09-22 03:22:43', 'active', 'yes', NULL, NULL),
(68, 'https://example.com/hsr/avatar44.png', 'hsr_trailblazer44@hsr.test', '', 'hsr_trailblazer44', 'March', 'Seven', '09280000044', 4, 'FACULTY', '2025-09-22 03:23:44', '2025-09-22 03:23:44', 'active', 'yes', NULL, NULL),
(69, 'https://example.com/hsr/avatar45.png', 'hsr_trailblazer45@hsr.test', '', 'hsr_trailblazer45', 'Kafka', 'Hunter', '09280000045', NULL, 'FACULTY', '2025-09-22 03:24:45', '2025-09-30 02:03:15', 'active', 'yes', NULL, NULL),
(70, 'https://example.com/hsr/avatar46.png', 'hsr_trailblazer46@hsr.test', '', 'hsr_trailblazer46', 'Silver', 'Wolf', '09280000046', 1, 'FACULTY', '2025-09-22 03:25:46', '2025-09-22 03:25:46', 'active', 'yes', NULL, NULL),
(71, 'https://example.com/hsr/avatar47.png', 'hsr_trailblazer47@hsr.test', '', 'hsr_trailblazer47', 'Clara', 'Stellaron', '09280000047', 2, 'FACULTY', '2025-09-22 03:26:47', '2025-09-22 03:26:47', 'active', 'yes', NULL, NULL),
(72, 'https://example.com/hsr/avatar48.png', 'hsr_trailblazer48@hsr.test', '', 'hsr_trailblazer48', 'Svarog', 'Guardian', '09280000048', 3, 'FACULTY', '2025-09-22 03:27:48', '2025-09-22 03:27:48', 'active', 'yes', NULL, NULL),
(73, 'https://example.com/hsr/avatar49.png', 'hsr_trailblazer49@hsr.test', '', 'hsr_trailblazer49', 'Himeko', 'Navigator', '09280000049', 4, 'FACULTY', '2025-09-22 03:28:49', '2025-09-22 03:28:49', 'active', 'yes', NULL, NULL),
(74, 'https://example.com/hsr/avatar50.png', 'hsr_trailblazer50@hsr.test', '', 'hsr_trailblazer50', 'Welt', 'Yang', '09280000050', NULL, 'FACULTY', '2025-09-22 03:29:50', '2025-09-30 02:03:15', 'active', 'yes', NULL, NULL),
(75, 'https://example.com/hsr/avatar51.png', 'hsr_trailblazer51@hsr.test', '', 'hsr_trailblazer51', 'Bronya', 'Rand', '09280000051', 2, 'DEAN', '2025-09-22 03:30:51', '2025-09-30 23:50:45', 'active', 'yes', NULL, NULL),
(76, 'https://example.com/hsr/avatar52.png', 'hsr_trailblazer52@hsr.test', '', 'hsr_trailblazer52', 'Seele', 'Butterfly', '09280000052', 2, 'FACULTY', '2025-09-22 03:31:52', '2025-09-22 03:31:52', 'active', 'yes', NULL, NULL),
(77, 'https://example.com/hsr/avatar53.png', 'hsr_trailblazer53@hsr.test', '', 'hsr_trailblazer53', 'Serval', 'Rockstar', '09280000053', 3, 'FACULTY', '2025-09-22 03:32:53', '2025-09-22 03:32:53', 'active', 'yes', NULL, NULL),
(78, 'https://example.com/hsr/avatar54.png', 'hsr_trailblazer54@hsr.test', '', 'hsr_trailblazer54', 'Natasha', 'Medic', '09280000054', 4, 'FACULTY', '2025-09-22 03:33:54', '2025-09-22 03:33:54', 'active', 'yes', NULL, NULL),
(79, 'https://example.com/hsr/avatar55.png', 'hsr_trailblazer55@hsr.test', '', 'hsr_trailblazer55', 'Pela', 'Analyst', '09280000055', NULL, 'FACULTY', '2025-09-22 03:34:55', '2025-09-30 02:03:15', 'active', 'yes', NULL, NULL),
(80, 'https://example.com/hsr/avatar56.png', 'hsr_trailblazer56@hsr.test', '', 'hsr_trailblazer56', 'Hook', 'Mole', '09280000056', 1, 'FACULTY', '2025-09-22 03:35:56', '2025-09-22 03:35:56', 'active', 'yes', NULL, NULL),
(81, 'https://example.com/hsr/avatar57.png', 'hsr_trailblazer57@hsr.test', '', 'hsr_trailblazer57', 'Sampo', 'Merchant', '09280000057', 2, 'FACULTY', '2025-09-22 03:36:57', '2025-09-22 03:36:57', 'active', 'yes', NULL, NULL),
(82, 'https://example.com/hsr/avatar58.png', 'hsr_trailblazer58@hsr.test', '', 'hsr_trailblazer58', 'Gepard', 'Knight', '09280000058', 3, 'FACULTY', '2025-09-22 03:37:58', '2025-09-22 03:37:58', 'active', 'yes', NULL, NULL),
(83, 'https://example.com/hsr/avatar59.png', 'hsr_trailblazer59@hsr.test', '', 'hsr_trailblazer59', 'Lynx', 'Explorer', '09280000059', 4, 'FACULTY', '2025-09-22 03:38:59', '2025-09-22 03:38:59', 'active', 'yes', NULL, NULL),
(84, 'https://example.com/hsr/avatar60.png', 'hsr_trailblazer60@hsr.test', '', 'hsr_trailblazer60', 'Luka', 'Boxer', '09280000060', NULL, 'FACULTY', '0000-00-00 00:00:00', '2025-09-30 02:03:15', 'active', 'yes', NULL, NULL),
(85, 'https://example.com/hsr/avatar61.png', 'hsr_trailblazer61@hsr.test', '', 'hsr_trailblazer61', 'Yukong', 'Helm', '09280000061', 1, 'FACULTY', '0000-00-00 00:00:00', '0000-00-00 00:00:00', 'active', 'yes', NULL, NULL),
(86, 'https://example.com/hsr/avatar62.png', 'hsr_trailblazer62@hsr.test', '', 'hsr_trailblazer62', 'Qingque', 'Gambler', '09280000062', 2, 'FACULTY', '0000-00-00 00:00:00', '0000-00-00 00:00:00', 'active', 'yes', NULL, NULL),
(87, 'https://example.com/hsr/avatar63.png', 'hsr_trailblazer63@hsr.test', '', 'hsr_trailblazer63', 'Tingyun', 'Broker', '09280000063', 3, 'FACULTY', '0000-00-00 00:00:00', '0000-00-00 00:00:00', 'active', 'yes', NULL, NULL),
(89, 'https://example.com/hsr/avatar65.png', 'hsr_trailblazer65@hsr.test', '', 'hsr_trailblazer65', 'Yan', 'Shou', '09280000065', NULL, 'FACULTY', '0000-00-00 00:00:00', '2025-09-30 02:03:15', 'active', 'yes', NULL, NULL),
(90, 'https://example.com/hsr/avatar66.png', 'hsr_trailblazer66@hsr.test', '', 'hsr_trailblazer66', 'Huohuo', 'Exorcist', '09280000066', 1, 'FACULTY', '0000-00-00 00:00:00', '0000-00-00 00:00:00', 'active', 'yes', NULL, NULL),
(93, 'https://example.com/hsr/avatar69.png', 'hsr_trailblazer69@hsr.test', '', 'hsr_trailblazer69', 'Sushang', 'Knight', '09280000069', 4, 'FACULTY', '0000-00-00 00:00:00', '0000-00-00 00:00:00', 'active', 'yes', NULL, NULL),
(94, 'https://example.com/hsr/avatar70.png', 'hsr_trailblazer70@hsr.test', '', 'hsr_trailblazer70', 'Herta', 'Genius', '09280000070', NULL, 'FACULTY', '0000-00-00 00:00:00', '2025-09-30 02:03:15', 'active', 'yes', NULL, NULL),
(95, 'https://example.com/hsr/avatar71.png', 'hsr_trailblazer71@hsr.test', '', 'hsr_trailblazer71', 'Luocha', 'Merchant', '09280000071', 1, 'FACULTY', '0000-00-00 00:00:00', '0000-00-00 00:00:00', 'active', 'yes', NULL, NULL),
(96, 'https://example.com/hsr/avatar72.png', 'hsr_trailblazer72@hsr.test', '', 'hsr_trailblazer72', 'Imbibitor', 'Lunae', '09280000072', 2, 'FACULTY', '0000-00-00 00:00:00', '0000-00-00 00:00:00', 'active', 'yes', NULL, NULL),
(97, 'https://example.com/hsr/avatar73.png', 'hsr_trailblazer73@hsr.test', '', 'hsr_trailblazer73', 'Dan', 'Shu', '09280000073', 3, 'FACULTY', '0000-00-00 00:00:00', '0000-00-00 00:00:00', 'active', 'yes', NULL, NULL),
(98, 'https://example.com/hsr/avatar74.png', 'hsr_trailblazer74@hsr.test', '', 'hsr_trailblazer74', 'Phantom', 'Edge', '09280000074', 4, 'FACULTY', '0000-00-00 00:00:00', '0000-00-00 00:00:00', 'active', 'yes', NULL, NULL),
(101, 'https://lh3.googleusercontent.com/a/ACg8ocLEtatCeMtXksrYW6U6JJqP0yRA92tumSUftqgGPHQuMyX53FA=s96-c', 'victoriyaklauss03@gmail.com', '', 'Victoriya Visha Von Klauss', 'sdf', 'sdf', '0918111', 2, 'DEAN', '2025-09-23 02:26:00', '2025-09-27 02:20:08', 'active', 'yes', NULL, NULL),
(102, 'https://lh3.googleusercontent.com/a/ACg8ocLV-XDgqUWjRyS9kVC_pJNkjOG7n6LJVcVeD7mWS9FDJ30CerUs=s96-c', 'mainlan03@gmail.com', '', 'Admin Lan', '', '', NULL, 4, 'DEAN', '2025-09-23 03:18:22', '2025-09-23 23:25:25', 'active', 'yes', NULL, NULL),
(103, 'https://lh3.googleusercontent.com/a/ACg8ocLEV9BcUyl_aJtC5tVHhJAdKUc32XVAM2W9Erx0sccP4R_S3g=s96-c', 'ispscladera@gmail.com', '$2b$12$8FEgyP7SjKKEpLZsfeyfC.jSDcVQZi/juKCYqF7yuBXQ0vjExZLjC', 'TEST1', 'cas', 'member', '09184226085', 1, 'FACULTY', '2025-09-30 23:46:25', '2025-10-01 00:41:46', 'active', 'yes', NULL, NULL);

--
-- Triggers `dms_user`
--
DELIMITER $$
CREATE TRIGGER `set_first_user_as_admin` BEFORE INSERT ON `dms_user` FOR EACH ROW BEGIN
    DECLARE user_count INT DEFAULT 0;
    
    -- Check if this is the first user being created
    SELECT COUNT(*) INTO user_count FROM dms_user;
    
    -- If this is the first user (user_count = 0), make them ADMIN
    IF user_count = 0 THEN
        SET NEW.role = 'ADMIN';
        SET NEW.is_verified = 'yes';
        SET NEW.status = 'active';
    END IF;
END
$$
DELIMITER ;
DELIMITER $$
CREATE TRIGGER `set_initial_user_status` BEFORE INSERT ON `dms_user` FOR EACH ROW BEGIN
    IF NEW.is_verified = 'yes' THEN
        SET NEW.status = 'active';
    ELSE
        SET NEW.status = 'pending';
    END IF;
END
$$
DELIMITER ;
DELIMITER $$
CREATE TRIGGER `update_user_status_on_verification` BEFORE UPDATE ON `dms_user` FOR EACH ROW BEGIN
    -- Only update status if is_verified is being changed and status is not being explicitly set
    IF NEW.is_verified != OLD.is_verified THEN
        IF NEW.is_verified = 'yes' AND OLD.is_verified = 'no' THEN
            SET NEW.status = 'active';
        ELSEIF NEW.is_verified = 'no' AND OLD.is_verified = 'yes' THEN
            SET NEW.status = 'pending';
        END IF;
    END IF;
END
$$
DELIMITER ;

-- --------------------------------------------------------

--
-- Table structure for table `document_actions`
--

CREATE TABLE `document_actions` (
  `document_action_id` int(11) NOT NULL,
  `doc_id` int(11) NOT NULL,
  `action_id` int(11) NOT NULL,
  `assigned_to_user_id` int(11) DEFAULT NULL,
  `assigned_to_role` enum('ADMIN','DEAN','FACULTY') DEFAULT NULL,
  `assigned_to_department_id` int(11) DEFAULT NULL,
  `status` enum('pending','in_progress','completed','cancelled') DEFAULT 'pending',
  `priority` enum('low','medium','high','urgent') DEFAULT 'medium',
  `due_date` date DEFAULT NULL,
  `completed_at` datetime DEFAULT NULL,
  `completed_by_user_id` int(11) DEFAULT NULL,
  `notes` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `created_by_user_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `document_actions`
--

INSERT INTO `document_actions` (`document_action_id`, `doc_id`, `action_id`, `assigned_to_user_id`, `assigned_to_role`, `assigned_to_department_id`, `status`, `priority`, `due_date`, `completed_at`, `completed_by_user_id`, `notes`, `created_at`, `updated_at`, `created_by_user_id`) VALUES
(11, 100, 1, 11, NULL, NULL, 'completed', 'medium', NULL, '2025-10-01 18:13:17', 1, NULL, '2025-09-30 21:13:45', '2025-10-01 10:13:17', 1);

-- --------------------------------------------------------

--
-- Table structure for table `document_departments`
--

CREATE TABLE `document_departments` (
  `doc_id` int(11) NOT NULL,
  `department_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `document_folders`
--

CREATE TABLE `document_folders` (
  `id` int(11) NOT NULL,
  `doc_id` int(11) NOT NULL,
  `folder_id` int(11) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `document_folders`
--

INSERT INTO `document_folders` (`id`, `doc_id`, `folder_id`, `created_at`) VALUES
(28, 96, 13, '2025-09-30 02:28:52'),
(29, 96, 14, '2025-09-30 02:28:52'),
(30, 96, 1, '2025-09-30 02:28:52'),
(31, 96, 8, '2025-09-30 02:28:52'),
(32, 96, 4, '2025-09-30 02:28:52'),
(33, 97, 999, '2025-09-30 02:41:01'),
(34, 99, 5, '2025-09-30 20:32:21'),
(35, 99, 10, '2025-09-30 20:37:42'),
(36, 99, 7, '2025-09-30 20:37:50'),
(37, 100, 5, '2025-09-30 21:13:45'),
(38, 101, 8, '2025-10-01 08:38:48');

-- --------------------------------------------------------

--
-- Table structure for table `document_roles`
--

CREATE TABLE `document_roles` (
  `doc_id` int(11) NOT NULL,
  `role` enum('ADMIN','DEAN','FACULTY') NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `document_types`
--

CREATE TABLE `document_types` (
  `type_id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `description` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `document_types`
--

INSERT INTO `document_types` (`type_id`, `name`, `description`, `created_at`, `updated_at`) VALUES
(5, 'Letter', NULL, '2025-09-29 00:02:46', '2025-09-29 00:02:46'),
(999, 'Unassigned', 'Default type for documents with deleted document types', '2025-09-30 02:34:44', '2025-09-30 02:34:44');

-- --------------------------------------------------------

--
-- Table structure for table `document_users`
--

CREATE TABLE `document_users` (
  `doc_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `folders`
--

CREATE TABLE `folders` (
  `folder_id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `folders`
--

INSERT INTO `folders` (`folder_id`, `name`, `created_at`, `updated_at`) VALUES
(4, 'Human Resources', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(5, 'Student Affairs', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(6, 'Faculty Development', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(7, 'Research and Extension', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(8, 'Administrative Orders', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(10, 'Procurement', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(11, 'Legal Documents', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(12, 'Archived Records', '2025-09-23 01:48:31', '2025-09-23 01:48:31'),
(999, 'Unassigned', '2025-09-30 02:34:44', '2025-09-30 02:34:44');

-- --------------------------------------------------------

--
-- Table structure for table `notifications`
--

CREATE TABLE `notifications` (
  `notification_id` int(11) NOT NULL,
  `title` varchar(255) NOT NULL,
  `message` text NOT NULL,
  `type` enum('added','updated','deleted','requested','replied') NOT NULL,
  `visible_to_all` tinyint(1) DEFAULT 1,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `related_doc_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `notifications`
--

INSERT INTO `notifications` (`notification_id`, `title`, `message`, `type`, `visible_to_all`, `created_at`, `related_doc_id`) VALUES
(22, 'New Announcement: ALL MEMBERS', 'A new announcement \"ALL MEMBERS\" has been created by ADMIN LAN', '', 1, '2025-09-07 09:15:50', NULL),
(24, 'New Announcement: ALL', 'A new announcement \"ALL\" has been created by ADMIN LAN', '', 1, '2025-09-07 10:09:55', NULL),
(25, 'New Announcement: again', 'A new announcement \"again\" has been created by ADMIN LAN', '', 0, '2025-09-07 10:12:18', NULL),
(26, 'New Announcement: wqe', 'A new announcement \"wqe\" has been created by ADMIN LAN', '', 1, '2025-09-07 11:56:47', NULL),
(27, 'New Announcement: das', 'A new announcement \"das\" has been created by ADMIN LAN', '', 1, '2025-09-07 11:58:20', NULL),
(28, 'New Announcement: das', 'A new announcement \"das\" has been created by ADMIN LAN', '', 0, '2025-09-07 12:01:44', NULL),
(29, 'New Announcement: das', 'A new announcement \"das\" has been created by ADMIN LAN', '', 1, '2025-09-07 12:02:03', NULL),
(299, 'Document Permanently Deleted', 'Document \"213\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(300, 'Document Permanently Deleted', 'Document \"3 Depeaty\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(301, 'Document Permanently Deleted', 'Document \"asd\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(302, 'Document Permanently Deleted', 'Document \"asd\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(303, 'Document Permanently Deleted', 'Document \"asda\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(304, 'Document Permanently Deleted', 'Document \"ASDAS\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(305, 'Document Permanently Deleted', 'Document \"Asdasdaaaaa\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(306, 'Document Permanently Deleted', 'Document \"CBMEzz22\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(307, 'Document Permanently Deleted', 'Document \"Certificate: Archive Completion\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(308, 'Document Permanently Deleted', 'Document \"Certificate: CAS Faculty Recognition\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(309, 'Document Permanently Deleted', 'Document \"Clearance: CAS Laboratory Equipment\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(310, 'Document Permanently Deleted', 'Document \"Clearance: Student Organization Renewal\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(311, 'Document Permanently Deleted', 'Document \"FOR ALL y\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(312, 'Document Permanently Deleted', 'Document \"FOR LEEVII - Link 1\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(313, 'Document Permanently Deleted', 'Document \"FOR LEEVII - Link 2\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(314, 'Document Permanently Deleted', 'Document \"FOR LEEVII - Link 3\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(315, 'Document Permanently Deleted', 'Document \"FOR LEVV1\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(316, 'Document Permanently Deleted', 'Document \"for ONLY ONE\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(317, 'Document Permanently Deleted', 'Document \"FOR USER YOU ONLY\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(318, 'Document Permanently Deleted', 'Document \"Guidelines: CAS Research Ethics\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(319, 'Document Permanently Deleted', 'Document \"Guidelines: Procurement Process\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(320, 'Document Permanently Deleted', 'Document \"Letter: Faculty Development Training Invite\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(321, 'Document Permanently Deleted', 'Document \"Letter: Invitation to CAS Academic Forum\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(322, 'Document Permanently Deleted', 'Document \"Memo: Faculty Meeting Notice\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(323, 'Document Permanently Deleted', 'Document \"New Document - Request 4 Vic\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(324, 'Document Permanently Deleted', 'Document \"Notice: CAS Exam Schedule Release\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(325, 'Document Permanently Deleted', 'Document \"Notice: Research Proposal Deadline\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(326, 'Document Permanently Deleted', 'Document \"please notify for levvi only\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(327, 'Document Permanently Deleted', 'Document \"pol\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(328, 'Document Permanently Deleted', 'Document \"Policy: CAS Faculty Evaluation\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(329, 'Document Permanently Deleted', 'Document \"Policy: Contract Management\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(330, 'Document Permanently Deleted', 'Document \"Proposal: CAS Library Expansion\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(331, 'Document Permanently Deleted', 'Document \"Proposal: Digital Filing System\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(332, 'Document Permanently Deleted', 'Document \"Report: Faculty Attendance Summary\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(333, 'Document Permanently Deleted', 'Document \"Report: Research Activities CAS\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(334, 'Document Permanently Deleted', 'Document \"Request 4 Vic\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:46', NULL),
(335, 'Document Permanently Deleted', 'Document \"Request: Additional Budget for Supplies\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:47', NULL),
(336, 'Document Permanently Deleted', 'Document \"Resolution: Admin Policy Update\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:47', NULL),
(337, 'Document Permanently Deleted', 'Document \"Resolution: CAS Curriculum Update\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:47', NULL),
(338, 'Document Permanently Deleted', 'Document \"sadas adpol\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:47', NULL),
(339, 'Document Permanently Deleted', 'Document \"sdf\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:47', NULL),
(340, 'Document Permanently Deleted', 'Document \"teesy\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:47', NULL),
(341, 'Document Permanently Deleted', 'Document \"TESINT - Link 1\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:47', NULL),
(342, 'Document Permanently Deleted', 'Document \"TESINT - Link 2\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:47', NULL),
(343, 'Document Permanently Deleted', 'Document \"TESINT - Link 3\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:47', NULL),
(344, 'Document Permanently Deleted', 'Document \"TESINT - Link 4\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:47', NULL),
(345, 'Document Permanently Deleted', 'Document \"TESINT - Link 5\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:47', NULL),
(346, 'Document Permanently Deleted', 'Document \"TEST 1\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:47', NULL),
(347, 'Document Permanently Deleted', 'Document \"TRE\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:43:47', NULL),
(351, 'Document Permanently Deleted', 'Document \"ASDASD\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:47:41', NULL),
(352, 'Document Permanently Deleted', 'Document \"assa\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:47:41', NULL),
(353, 'Document Permanently Deleted', 'Document \"sasasasas\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 00:47:41', NULL),
(356, 'Document Permanently Deleted', 'Document \"Test\" has been permanently deleted from trashcan', 'deleted', 1, '2025-09-30 02:35:26', NULL),
(361, 'New Document Added: sdds', 'A new document \"sdds\" has been uploaded by ADMIN IS MEa', 'added', 1, '2025-09-30 20:32:21', 99),
(363, 'Request Added: asd', 'You have a new request \"asd\" from ADMINaa', 'requested', 0, '2025-09-30 21:13:45', 100),
(367, 'New Announcement: asd', 'asd', '', 1, '2025-09-30 21:40:06', NULL),
(368, 'New Announcement: TEST 1 FOR ALL', '123312', '', 0, '2025-09-30 21:40:43', NULL),
(369, 'New Announcement: swer', 'ewr', '', 0, '2025-09-30 23:14:38', NULL),
(371, 'New User Registration: cas member', 'A new user \"TEST1\" (ispscladera@gmail.com) has registered via signup form.', '', 0, '2025-09-30 23:46:25', NULL),
(372, 'Document Permanently Deleted', 'Document \"Lll\" has been permanently deleted from trashcan', 'deleted', 1, '2025-10-01 07:52:26', NULL),
(373, 'Document Permanently Deleted', 'Document \"Aaaa\" has been permanently deleted from trashcan', 'deleted', 1, '2025-10-01 07:52:33', NULL),
(374, 'New Document Added: MOBILE TEST', 'A new document \"MOBILE TEST\" has been uploaded by THIS IS ME', 'added', 1, '2025-10-01 08:38:48', 101),
(375, 'Reply Added: asdas', 'ADMIN LAN LAN replied to your request \"asdas\".', 'replied', 0, '2025-10-01 10:13:17', 100);

-- --------------------------------------------------------

--
-- Table structure for table `notification_departments`
--

CREATE TABLE `notification_departments` (
  `notification_id` int(11) NOT NULL,
  `department_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `notification_departments`
--

INSERT INTO `notification_departments` (`notification_id`, `department_id`) VALUES
(25, 1),
(28, 1),
(368, 2),
(369, 1);

-- --------------------------------------------------------

--
-- Table structure for table `notification_reads`
--

CREATE TABLE `notification_reads` (
  `notification_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `read_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `notification_reads`
--

INSERT INTO `notification_reads` (`notification_id`, `user_id`, `read_at`) VALUES
(22, 1, '2025-10-01 00:48:25'),
(22, 3, '2025-09-30 00:50:29'),
(22, 9, '2025-09-30 21:35:19'),
(22, 11, '2025-09-23 09:41:09'),
(22, 101, '2025-09-30 21:34:10'),
(22, 103, '2025-09-30 23:48:23'),
(24, 1, '2025-10-01 00:48:25'),
(24, 3, '2025-09-30 00:50:29'),
(24, 9, '2025-09-30 21:35:19'),
(24, 11, '2025-09-23 09:41:09'),
(24, 101, '2025-09-30 21:34:10'),
(24, 103, '2025-09-30 23:48:23'),
(25, 1, '2025-10-01 00:48:25'),
(25, 9, '2025-09-30 21:35:19'),
(25, 11, '2025-09-23 09:41:09'),
(25, 103, '2025-09-30 23:48:23'),
(26, 1, '2025-10-01 00:48:25'),
(26, 3, '2025-09-30 00:50:29'),
(26, 9, '2025-09-30 21:35:19'),
(26, 11, '2025-09-23 09:41:09'),
(26, 101, '2025-09-30 21:34:10'),
(26, 103, '2025-09-30 23:48:23'),
(27, 1, '2025-10-01 00:48:25'),
(27, 3, '2025-09-30 00:50:29'),
(27, 9, '2025-09-30 21:35:19'),
(27, 11, '2025-09-23 09:41:09'),
(27, 101, '2025-09-30 21:34:10'),
(27, 103, '2025-09-30 23:48:23'),
(28, 1, '2025-10-01 00:48:25'),
(28, 9, '2025-09-30 21:35:19'),
(28, 103, '2025-09-30 23:54:39'),
(29, 1, '2025-10-01 00:48:25'),
(29, 3, '2025-09-30 00:50:29'),
(29, 9, '2025-09-30 21:35:19'),
(29, 11, '2025-09-23 09:41:09'),
(29, 101, '2025-09-30 21:34:10'),
(29, 103, '2025-09-30 23:48:23'),
(299, 1, '2025-10-01 00:48:25'),
(299, 3, '2025-09-30 00:50:29'),
(299, 9, '2025-09-30 21:35:19'),
(299, 101, '2025-09-30 21:34:10'),
(299, 103, '2025-09-30 23:48:23'),
(300, 1, '2025-10-01 00:48:25'),
(300, 3, '2025-09-30 00:50:29'),
(300, 9, '2025-09-30 21:35:19'),
(300, 101, '2025-09-30 21:34:10'),
(300, 103, '2025-09-30 23:48:23'),
(301, 1, '2025-10-01 00:48:25'),
(301, 3, '2025-09-30 00:50:29'),
(301, 9, '2025-09-30 21:35:19'),
(301, 101, '2025-09-30 21:34:10'),
(301, 103, '2025-09-30 23:48:23'),
(302, 1, '2025-10-01 00:48:25'),
(302, 3, '2025-09-30 00:50:29'),
(302, 9, '2025-09-30 21:35:19'),
(302, 101, '2025-09-30 21:34:10'),
(302, 103, '2025-09-30 23:48:23'),
(303, 1, '2025-10-01 00:48:25'),
(303, 3, '2025-09-30 00:50:29'),
(303, 9, '2025-09-30 21:35:19'),
(303, 101, '2025-09-30 21:34:10'),
(303, 103, '2025-09-30 23:48:23'),
(304, 1, '2025-10-01 00:48:25'),
(304, 3, '2025-09-30 00:50:29'),
(304, 9, '2025-09-30 21:35:19'),
(304, 101, '2025-09-30 21:34:10'),
(304, 103, '2025-09-30 23:48:23'),
(305, 1, '2025-10-01 00:48:25'),
(305, 3, '2025-09-30 00:50:29'),
(305, 9, '2025-09-30 21:35:19'),
(305, 101, '2025-09-30 21:34:10'),
(305, 103, '2025-09-30 23:48:23'),
(306, 1, '2025-10-01 00:48:25'),
(306, 3, '2025-09-30 00:50:29'),
(306, 9, '2025-09-30 21:35:19'),
(306, 101, '2025-09-30 21:34:10'),
(306, 103, '2025-09-30 23:48:23'),
(307, 1, '2025-10-01 00:48:25'),
(307, 3, '2025-09-30 00:50:29'),
(307, 9, '2025-09-30 21:35:19'),
(307, 101, '2025-09-30 21:34:10'),
(307, 103, '2025-09-30 23:48:23'),
(308, 1, '2025-10-01 00:48:25'),
(308, 3, '2025-09-30 00:50:29'),
(308, 9, '2025-09-30 21:35:19'),
(308, 101, '2025-09-30 21:34:10'),
(308, 103, '2025-09-30 23:48:23'),
(309, 1, '2025-10-01 00:48:25'),
(309, 3, '2025-09-30 00:50:29'),
(309, 9, '2025-09-30 21:35:19'),
(309, 101, '2025-09-30 21:34:10'),
(309, 103, '2025-09-30 23:48:23'),
(310, 1, '2025-10-01 00:48:25'),
(310, 3, '2025-09-30 00:50:29'),
(310, 9, '2025-09-30 21:35:19'),
(310, 101, '2025-09-30 21:34:10'),
(310, 103, '2025-09-30 23:48:23'),
(311, 1, '2025-10-01 00:48:25'),
(311, 3, '2025-09-30 00:50:29'),
(311, 9, '2025-09-30 21:35:19'),
(311, 101, '2025-09-30 21:34:10'),
(311, 103, '2025-09-30 23:48:23'),
(312, 1, '2025-10-01 00:48:25'),
(312, 3, '2025-09-30 00:50:29'),
(312, 9, '2025-09-30 21:35:19'),
(312, 101, '2025-09-30 21:34:10'),
(312, 103, '2025-09-30 23:48:23'),
(313, 1, '2025-10-01 00:48:25'),
(313, 3, '2025-09-30 00:50:29'),
(313, 9, '2025-09-30 21:35:19'),
(313, 101, '2025-09-30 21:34:10'),
(313, 103, '2025-09-30 23:48:23'),
(314, 1, '2025-10-01 00:48:25'),
(314, 3, '2025-09-30 00:50:29'),
(314, 9, '2025-09-30 21:35:19'),
(314, 101, '2025-09-30 21:34:10'),
(314, 103, '2025-09-30 23:48:23'),
(315, 1, '2025-10-01 00:48:25'),
(315, 3, '2025-09-30 00:50:29'),
(315, 9, '2025-09-30 21:35:19'),
(315, 101, '2025-09-30 21:34:10'),
(315, 103, '2025-09-30 23:48:23'),
(316, 1, '2025-10-01 00:48:25'),
(316, 3, '2025-09-30 00:50:29'),
(316, 9, '2025-09-30 21:35:19'),
(316, 101, '2025-09-30 21:34:10'),
(316, 103, '2025-09-30 23:48:23'),
(317, 1, '2025-10-01 00:48:25'),
(317, 3, '2025-09-30 00:50:29'),
(317, 9, '2025-09-30 21:35:19'),
(317, 101, '2025-09-30 21:34:10'),
(317, 103, '2025-09-30 23:48:23'),
(318, 1, '2025-10-01 00:48:25'),
(318, 3, '2025-09-30 00:50:29'),
(318, 9, '2025-09-30 21:35:19'),
(318, 101, '2025-09-30 21:34:10'),
(318, 103, '2025-09-30 23:48:23'),
(319, 1, '2025-10-01 00:48:25'),
(319, 3, '2025-09-30 00:50:29'),
(319, 9, '2025-09-30 21:35:19'),
(319, 101, '2025-09-30 21:34:10'),
(319, 103, '2025-09-30 23:48:23'),
(320, 1, '2025-10-01 00:48:25'),
(320, 3, '2025-09-30 00:50:29'),
(320, 9, '2025-09-30 21:35:19'),
(320, 101, '2025-09-30 21:34:10'),
(320, 103, '2025-09-30 23:48:23'),
(321, 1, '2025-10-01 00:48:25'),
(321, 3, '2025-09-30 00:50:29'),
(321, 9, '2025-09-30 21:35:19'),
(321, 101, '2025-09-30 21:34:10'),
(321, 103, '2025-09-30 23:48:23'),
(322, 1, '2025-10-01 00:48:25'),
(322, 3, '2025-09-30 00:50:29'),
(322, 9, '2025-09-30 21:35:19'),
(322, 101, '2025-09-30 21:34:10'),
(322, 103, '2025-09-30 23:48:23'),
(323, 1, '2025-10-01 00:48:25'),
(323, 3, '2025-09-30 00:50:29'),
(323, 9, '2025-09-30 21:35:19'),
(323, 101, '2025-09-30 21:34:10'),
(323, 103, '2025-09-30 23:48:23'),
(324, 1, '2025-10-01 00:48:25'),
(324, 3, '2025-09-30 00:50:29'),
(324, 9, '2025-09-30 21:35:19'),
(324, 101, '2025-09-30 21:34:10'),
(324, 103, '2025-09-30 23:48:23'),
(325, 1, '2025-10-01 00:48:25'),
(325, 3, '2025-09-30 00:50:29'),
(325, 9, '2025-09-30 21:35:19'),
(325, 101, '2025-09-30 21:34:10'),
(325, 103, '2025-09-30 23:48:23'),
(326, 1, '2025-10-01 00:48:25'),
(326, 3, '2025-09-30 00:50:29'),
(326, 9, '2025-09-30 21:35:19'),
(326, 101, '2025-09-30 21:34:10'),
(326, 103, '2025-09-30 23:48:23'),
(327, 1, '2025-10-01 00:48:25'),
(327, 3, '2025-09-30 00:50:29'),
(327, 9, '2025-09-30 21:35:19'),
(327, 101, '2025-09-30 21:34:10'),
(327, 103, '2025-09-30 23:48:23'),
(328, 1, '2025-10-01 00:48:25'),
(328, 3, '2025-09-30 00:50:29'),
(328, 9, '2025-09-30 21:35:19'),
(328, 101, '2025-09-30 21:34:10'),
(328, 103, '2025-09-30 23:48:23'),
(329, 1, '2025-10-01 00:48:25'),
(329, 3, '2025-09-30 00:50:29'),
(329, 9, '2025-09-30 21:35:19'),
(329, 101, '2025-09-30 21:34:10'),
(329, 103, '2025-09-30 23:48:23'),
(330, 1, '2025-10-01 00:48:25'),
(330, 3, '2025-09-30 00:50:29'),
(330, 9, '2025-09-30 21:35:19'),
(330, 101, '2025-09-30 21:34:10'),
(330, 103, '2025-09-30 23:48:23'),
(331, 1, '2025-10-01 00:48:25'),
(331, 3, '2025-09-30 00:50:29'),
(331, 9, '2025-09-30 21:35:19'),
(331, 101, '2025-09-30 21:34:10'),
(331, 103, '2025-09-30 23:48:23'),
(332, 1, '2025-10-01 00:48:25'),
(332, 3, '2025-09-30 00:50:29'),
(332, 9, '2025-09-30 21:35:19'),
(332, 101, '2025-09-30 21:34:10'),
(332, 103, '2025-09-30 23:48:23'),
(333, 1, '2025-10-01 00:48:25'),
(333, 3, '2025-09-30 00:50:29'),
(333, 9, '2025-09-30 21:35:19'),
(333, 101, '2025-09-30 21:34:10'),
(333, 103, '2025-09-30 23:48:23'),
(334, 1, '2025-10-01 00:48:25'),
(334, 3, '2025-09-30 00:50:29'),
(334, 9, '2025-09-30 21:35:19'),
(334, 101, '2025-09-30 21:34:10'),
(334, 103, '2025-09-30 23:48:23'),
(335, 1, '2025-10-01 00:48:25'),
(335, 3, '2025-09-30 00:50:29'),
(335, 9, '2025-09-30 21:35:19'),
(335, 101, '2025-09-30 21:34:10'),
(335, 103, '2025-09-30 23:48:23'),
(336, 1, '2025-10-01 00:48:25'),
(336, 3, '2025-09-30 00:50:29'),
(336, 9, '2025-09-30 21:35:19'),
(336, 101, '2025-09-30 21:34:10'),
(336, 103, '2025-09-30 23:48:23'),
(337, 1, '2025-10-01 00:48:25'),
(337, 3, '2025-09-30 00:50:29'),
(337, 9, '2025-09-30 21:35:19'),
(337, 101, '2025-09-30 21:34:10'),
(337, 103, '2025-09-30 23:48:23'),
(338, 1, '2025-10-01 00:48:25'),
(338, 3, '2025-09-30 00:50:29'),
(338, 9, '2025-09-30 21:35:19'),
(338, 101, '2025-09-30 21:34:10'),
(338, 103, '2025-09-30 23:48:23'),
(339, 1, '2025-10-01 00:48:25'),
(339, 3, '2025-09-30 00:50:29'),
(339, 9, '2025-09-30 21:35:19'),
(339, 101, '2025-09-30 21:34:10'),
(339, 103, '2025-09-30 23:48:23'),
(340, 1, '2025-10-01 00:48:25'),
(340, 3, '2025-09-30 00:50:29'),
(340, 9, '2025-09-30 21:35:19'),
(340, 101, '2025-09-30 21:34:10'),
(340, 103, '2025-09-30 23:48:23'),
(341, 1, '2025-10-01 00:48:25'),
(341, 3, '2025-09-30 00:50:29'),
(341, 9, '2025-09-30 21:35:19'),
(341, 101, '2025-09-30 21:34:10'),
(341, 103, '2025-09-30 23:48:23'),
(342, 1, '2025-10-01 00:48:25'),
(342, 3, '2025-09-30 00:50:29'),
(342, 9, '2025-09-30 21:35:19'),
(342, 101, '2025-09-30 21:34:10'),
(342, 103, '2025-09-30 23:48:23'),
(343, 1, '2025-10-01 00:48:25'),
(343, 3, '2025-09-30 00:50:29'),
(343, 9, '2025-09-30 21:35:19'),
(343, 101, '2025-09-30 21:34:10'),
(343, 103, '2025-09-30 23:48:23'),
(344, 1, '2025-10-01 00:48:25'),
(344, 3, '2025-09-30 00:50:29'),
(344, 9, '2025-09-30 21:35:19'),
(344, 101, '2025-09-30 21:34:10'),
(344, 103, '2025-09-30 23:48:23'),
(345, 1, '2025-10-01 00:48:25'),
(345, 3, '2025-09-30 00:50:29'),
(345, 9, '2025-09-30 21:35:19'),
(345, 101, '2025-09-30 21:34:10'),
(345, 103, '2025-09-30 23:48:23'),
(346, 1, '2025-10-01 00:48:25'),
(346, 3, '2025-09-30 00:50:29'),
(346, 9, '2025-09-30 21:35:19'),
(346, 101, '2025-09-30 21:34:10'),
(346, 103, '2025-09-30 23:48:23'),
(347, 1, '2025-10-01 00:48:25'),
(347, 3, '2025-09-30 00:50:29'),
(347, 9, '2025-09-30 21:35:19'),
(347, 101, '2025-09-30 21:34:10'),
(347, 103, '2025-09-30 23:48:23'),
(351, 1, '2025-10-01 00:48:25'),
(351, 3, '2025-09-30 00:50:29'),
(351, 9, '2025-09-30 21:35:19'),
(351, 101, '2025-09-30 21:34:10'),
(351, 103, '2025-09-30 23:48:23'),
(352, 1, '2025-10-01 00:48:25'),
(352, 3, '2025-09-30 00:50:29'),
(352, 9, '2025-09-30 21:35:19'),
(352, 101, '2025-09-30 21:34:10'),
(352, 103, '2025-09-30 23:48:23'),
(353, 1, '2025-10-01 00:48:25'),
(353, 3, '2025-09-30 00:50:29'),
(353, 9, '2025-09-30 21:35:19'),
(353, 101, '2025-09-30 21:34:10'),
(353, 103, '2025-09-30 23:48:23'),
(356, 1, '2025-10-01 00:48:25'),
(356, 9, '2025-09-30 21:35:19'),
(356, 101, '2025-09-30 21:34:10'),
(356, 103, '2025-09-30 23:48:23'),
(361, 1, '2025-10-01 00:48:25'),
(361, 9, '2025-09-30 21:35:19'),
(361, 101, '2025-09-30 21:34:10'),
(361, 103, '2025-09-30 23:48:23'),
(367, 1, '2025-10-01 00:48:25'),
(367, 101, '2025-09-30 23:45:13'),
(367, 103, '2025-09-30 23:48:23'),
(368, 1, '2025-10-01 00:48:25'),
(368, 101, '2025-09-30 23:45:13'),
(368, 103, '2025-09-30 23:48:23'),
(369, 1, '2025-10-01 00:48:25'),
(369, 103, '2025-09-30 23:54:39'),
(371, 1, '2025-10-01 00:48:25');

-- --------------------------------------------------------

--
-- Table structure for table `notification_roles`
--

CREATE TABLE `notification_roles` (
  `notification_id` int(11) NOT NULL,
  `role` enum('ADMIN','DEAN','FACULTY') NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `notification_roles`
--

INSERT INTO `notification_roles` (`notification_id`, `role`) VALUES
(25, 'FACULTY'),
(368, 'ADMIN'),
(368, 'DEAN'),
(368, 'FACULTY'),
(375, 'ADMIN');

-- --------------------------------------------------------

--
-- Table structure for table `notification_users`
--

CREATE TABLE `notification_users` (
  `notification_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `notification_users`
--

INSERT INTO `notification_users` (`notification_id`, `user_id`) VALUES
(363, 11),
(371, 1),
(371, 3),
(371, 101),
(371, 102),
(375, 11);

-- --------------------------------------------------------

--
-- Table structure for table `system_settings`
--

CREATE TABLE `system_settings` (
  `id` int(11) NOT NULL,
  `maintenance_mode` tinyint(1) NOT NULL DEFAULT 0,
  `maintenance_message` text DEFAULT NULL,
  `maintenance_start_time` datetime DEFAULT NULL,
  `maintenance_end_time` datetime DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `system_settings`
--

INSERT INTO `system_settings` (`id`, `maintenance_mode`, `maintenance_message`, `maintenance_start_time`, `maintenance_end_time`, `created_at`, `updated_at`) VALUES
(1, 0, 'Test', '2025-09-18 16:44:00', '2025-09-18 16:45:00', '2025-09-14 20:29:10', '2025-09-29 05:16:26');

-- --------------------------------------------------------

--
-- Table structure for table `user_document_preferences`
--

CREATE TABLE `user_document_preferences` (
  `preference_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `doc_id` int(11) NOT NULL,
  `is_favorite` tinyint(1) DEFAULT 0,
  `is_pinned` tinyint(1) DEFAULT 0,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `user_document_preferences`
--

INSERT INTO `user_document_preferences` (`preference_id`, `user_id`, `doc_id`, `is_favorite`, `is_pinned`, `created_at`, `updated_at`) VALUES
(16, 1, 99, 0, 0, '2025-09-30 20:33:03', '2025-10-01 08:38:53'),
(17, 103, 99, 1, 0, '2025-09-30 23:54:51', '2025-09-30 23:56:52'),
(18, 101, 99, 1, 1, '2025-09-30 23:57:39', '2025-10-01 00:22:19'),
(19, 11, 99, 1, 0, '2025-09-30 23:58:50', '2025-09-30 23:58:50'),
(21, 9, 99, 0, 0, '2025-10-01 00:14:04', '2025-10-01 00:22:37'),
(22, 1, 101, 1, 0, '2025-10-01 08:38:54', '2025-10-01 08:38:54');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `action_required`
--
ALTER TABLE `action_required`
  ADD PRIMARY KEY (`action_id`),
  ADD KEY `idx_action_category` (`action_category`),
  ADD KEY `idx_is_active` (`is_active`),
  ADD KEY `idx_created_by` (`created_by`);

--
-- Indexes for table `announcements`
--
ALTER TABLE `announcements`
  ADD PRIMARY KEY (`announcement_id`),
  ADD KEY `idx_status` (`status`),
  ADD KEY `idx_publish_at` (`publish_at`),
  ADD KEY `idx_expire_at` (`expire_at`);

--
-- Indexes for table `announcement_departments`
--
ALTER TABLE `announcement_departments`
  ADD PRIMARY KEY (`announcement_id`,`department_id`),
  ADD KEY `department_id` (`department_id`);

--
-- Indexes for table `announcement_roles`
--
ALTER TABLE `announcement_roles`
  ADD PRIMARY KEY (`announcement_id`,`role`),
  ADD KEY `role` (`role`);

--
-- Indexes for table `announcement_users`
--
ALTER TABLE `announcement_users`
  ADD PRIMARY KEY (`announcement_id`,`user_id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `departments`
--
ALTER TABLE `departments`
  ADD PRIMARY KEY (`department_id`),
  ADD UNIQUE KEY `name` (`name`),
  ADD UNIQUE KEY `code` (`code`);

--
-- Indexes for table `dms_documents`
--
ALTER TABLE `dms_documents`
  ADD PRIMARY KEY (`doc_id`),
  ADD KEY `idx_doc_type` (`doc_type`),
  ADD KEY `idx_title` (`title`),
  ADD KEY `idx_available_copy` (`available_copy`),
  ADD KEY `idx_received_by_user_id` (`received_by_user_id`),
  ADD KEY `idx_visible_to_all` (`visible_to_all`),
  ADD KEY `idx_status` (`status`),
  ADD KEY `idx_deleted` (`deleted`),
  ADD KEY `idx_deleted_at` (`deleted_at`),
  ADD KEY `idx_deleted_by_name` (`deleted_by_name`),
  ADD KEY `dms_documents_ibfk_reply` (`is_reply_to_doc_id`),
  ADD KEY `idx_folder_id` (`folder_id`),
  ADD KEY `dms_documents_ibfk_created_by` (`created_by_user_id`);

--
-- Indexes for table `dms_user`
--
ALTER TABLE `dms_user`
  ADD PRIMARY KEY (`user_id`),
  ADD KEY `dms_user_ibfk_department` (`department_id`);

--
-- Indexes for table `document_actions`
--
ALTER TABLE `document_actions`
  ADD PRIMARY KEY (`document_action_id`),
  ADD KEY `idx_doc_id` (`doc_id`),
  ADD KEY `idx_action_id` (`action_id`),
  ADD KEY `idx_assigned_to_user_id` (`assigned_to_user_id`),
  ADD KEY `idx_assigned_to_role` (`assigned_to_role`),
  ADD KEY `idx_assigned_to_department_id` (`assigned_to_department_id`),
  ADD KEY `idx_status` (`status`),
  ADD KEY `idx_priority` (`priority`),
  ADD KEY `idx_due_date` (`due_date`),
  ADD KEY `idx_completed_by_user_id` (`completed_by_user_id`),
  ADD KEY `idx_created_by_user_id` (`created_by_user_id`);

--
-- Indexes for table `document_departments`
--
ALTER TABLE `document_departments`
  ADD KEY `document_departments_ibfk_doc` (`doc_id`),
  ADD KEY `document_departments_ibfk_department` (`department_id`);

--
-- Indexes for table `document_folders`
--
ALTER TABLE `document_folders`
  ADD PRIMARY KEY (`id`),
  ADD KEY `idx_doc` (`doc_id`),
  ADD KEY `idx_folder` (`folder_id`);

--
-- Indexes for table `document_roles`
--
ALTER TABLE `document_roles`
  ADD KEY `document_roles_ibfk_doc` (`doc_id`);

--
-- Indexes for table `document_types`
--
ALTER TABLE `document_types`
  ADD PRIMARY KEY (`type_id`);

--
-- Indexes for table `document_users`
--
ALTER TABLE `document_users`
  ADD KEY `document_users_ibfk_doc` (`doc_id`),
  ADD KEY `document_users_ibfk_user` (`user_id`);

--
-- Indexes for table `folders`
--
ALTER TABLE `folders`
  ADD PRIMARY KEY (`folder_id`),
  ADD UNIQUE KEY `name` (`name`);

--
-- Indexes for table `notifications`
--
ALTER TABLE `notifications`
  ADD PRIMARY KEY (`notification_id`),
  ADD KEY `idx_type` (`type`),
  ADD KEY `idx_visible_to_all` (`visible_to_all`),
  ADD KEY `idx_created_at` (`created_at`),
  ADD KEY `idx_related_doc_id` (`related_doc_id`);

--
-- Indexes for table `notification_departments`
--
ALTER TABLE `notification_departments`
  ADD PRIMARY KEY (`notification_id`,`department_id`),
  ADD KEY `department_id` (`department_id`),
  ADD KEY `idx_nd_notif` (`notification_id`),
  ADD KEY `idx_nd_dept` (`department_id`);

--
-- Indexes for table `notification_reads`
--
ALTER TABLE `notification_reads`
  ADD PRIMARY KEY (`notification_id`,`user_id`),
  ADD UNIQUE KEY `uq_notification_reads` (`notification_id`,`user_id`),
  ADD KEY `idx_user_id` (`user_id`);

--
-- Indexes for table `notification_roles`
--
ALTER TABLE `notification_roles`
  ADD PRIMARY KEY (`notification_id`,`role`),
  ADD KEY `role` (`role`),
  ADD KEY `idx_nr_notif` (`notification_id`),
  ADD KEY `idx_nr_role` (`role`);

--
-- Indexes for table `notification_users`
--
ALTER TABLE `notification_users`
  ADD PRIMARY KEY (`notification_id`,`user_id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `idx_nu_notif` (`notification_id`),
  ADD KEY `idx_nu_user` (`user_id`);

--
-- Indexes for table `system_settings`
--
ALTER TABLE `system_settings`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `user_document_preferences`
--
ALTER TABLE `user_document_preferences`
  ADD PRIMARY KEY (`preference_id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `doc_id` (`doc_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `action_required`
--
ALTER TABLE `action_required`
  MODIFY `action_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=18;

--
-- AUTO_INCREMENT for table `announcements`
--
ALTER TABLE `announcements`
  MODIFY `announcement_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=31;

--
-- AUTO_INCREMENT for table `departments`
--
ALTER TABLE `departments`
  MODIFY `department_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1000;

--
-- AUTO_INCREMENT for table `dms_documents`
--
ALTER TABLE `dms_documents`
  MODIFY `doc_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=103;

--
-- AUTO_INCREMENT for table `dms_user`
--
ALTER TABLE `dms_user`
  MODIFY `user_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=104;

--
-- AUTO_INCREMENT for table `document_actions`
--
ALTER TABLE `document_actions`
  MODIFY `document_action_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=12;

--
-- AUTO_INCREMENT for table `document_folders`
--
ALTER TABLE `document_folders`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=39;

--
-- AUTO_INCREMENT for table `document_types`
--
ALTER TABLE `document_types`
  MODIFY `type_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1000;

--
-- AUTO_INCREMENT for table `folders`
--
ALTER TABLE `folders`
  MODIFY `folder_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1001;

--
-- AUTO_INCREMENT for table `notifications`
--
ALTER TABLE `notifications`
  MODIFY `notification_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=376;

--
-- AUTO_INCREMENT for table `user_document_preferences`
--
ALTER TABLE `user_document_preferences`
  MODIFY `preference_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=23;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `announcement_departments`
--
ALTER TABLE `announcement_departments`
  ADD CONSTRAINT `announcement_departments_ibfk_announcement` FOREIGN KEY (`announcement_id`) REFERENCES `announcements` (`announcement_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `announcement_departments_ibfk_department` FOREIGN KEY (`department_id`) REFERENCES `departments` (`department_id`) ON DELETE CASCADE;

--
-- Constraints for table `announcement_roles`
--
ALTER TABLE `announcement_roles`
  ADD CONSTRAINT `announcement_roles_ibfk_announcement` FOREIGN KEY (`announcement_id`) REFERENCES `announcements` (`announcement_id`) ON DELETE CASCADE;

--
-- Constraints for table `announcement_users`
--
ALTER TABLE `announcement_users`
  ADD CONSTRAINT `announcement_users_ibfk_announcement` FOREIGN KEY (`announcement_id`) REFERENCES `announcements` (`announcement_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `announcement_users_ibfk_user` FOREIGN KEY (`user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE CASCADE;

--
-- Constraints for table `dms_documents`
--
ALTER TABLE `dms_documents`
  ADD CONSTRAINT `dms_documents_ibfk_created_by` FOREIGN KEY (`created_by_user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `dms_documents_ibfk_doc_type` FOREIGN KEY (`doc_type`) REFERENCES `document_types` (`type_id`),
  ADD CONSTRAINT `dms_documents_ibfk_folder` FOREIGN KEY (`folder_id`) REFERENCES `folders` (`folder_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `dms_documents_ibfk_received_by` FOREIGN KEY (`received_by_user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `dms_documents_ibfk_reply` FOREIGN KEY (`is_reply_to_doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE SET NULL;

--
-- Constraints for table `dms_user`
--
ALTER TABLE `dms_user`
  ADD CONSTRAINT `dms_user_ibfk_department` FOREIGN KEY (`department_id`) REFERENCES `departments` (`department_id`) ON DELETE SET NULL;

--
-- Constraints for table `document_actions`
--
ALTER TABLE `document_actions`
  ADD CONSTRAINT `document_actions_ibfk_action` FOREIGN KEY (`action_id`) REFERENCES `action_required` (`action_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `document_actions_ibfk_assigned_department` FOREIGN KEY (`assigned_to_department_id`) REFERENCES `departments` (`department_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `document_actions_ibfk_assigned_user` FOREIGN KEY (`assigned_to_user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `document_actions_ibfk_completed_by` FOREIGN KEY (`completed_by_user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `document_actions_ibfk_created_by` FOREIGN KEY (`created_by_user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE SET NULL,
  ADD CONSTRAINT `document_actions_ibfk_doc` FOREIGN KEY (`doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE CASCADE;

--
-- Constraints for table `document_departments`
--
ALTER TABLE `document_departments`
  ADD CONSTRAINT `document_departments_ibfk_department` FOREIGN KEY (`department_id`) REFERENCES `departments` (`department_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `document_departments_ibfk_doc` FOREIGN KEY (`doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE CASCADE;

--
-- Constraints for table `document_roles`
--
ALTER TABLE `document_roles`
  ADD CONSTRAINT `document_roles_ibfk_doc` FOREIGN KEY (`doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE CASCADE;

--
-- Constraints for table `document_users`
--
ALTER TABLE `document_users`
  ADD CONSTRAINT `document_users_ibfk_doc` FOREIGN KEY (`doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `document_users_ibfk_user` FOREIGN KEY (`user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE CASCADE;

--
-- Constraints for table `notifications`
--
ALTER TABLE `notifications`
  ADD CONSTRAINT `notifications_ibfk_doc` FOREIGN KEY (`related_doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE CASCADE;

--
-- Constraints for table `notification_departments`
--
ALTER TABLE `notification_departments`
  ADD CONSTRAINT `notification_departments_ibfk_department` FOREIGN KEY (`department_id`) REFERENCES `departments` (`department_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `notification_departments_ibfk_notification` FOREIGN KEY (`notification_id`) REFERENCES `notifications` (`notification_id`) ON DELETE CASCADE;

--
-- Constraints for table `notification_reads`
--
ALTER TABLE `notification_reads`
  ADD CONSTRAINT `nr_fk_notification` FOREIGN KEY (`notification_id`) REFERENCES `notifications` (`notification_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `nr_fk_user` FOREIGN KEY (`user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE CASCADE;

--
-- Constraints for table `notification_roles`
--
ALTER TABLE `notification_roles`
  ADD CONSTRAINT `notification_roles_ibfk_notification` FOREIGN KEY (`notification_id`) REFERENCES `notifications` (`notification_id`) ON DELETE CASCADE;

--
-- Constraints for table `notification_users`
--
ALTER TABLE `notification_users`
  ADD CONSTRAINT `notification_users_ibfk_notification` FOREIGN KEY (`notification_id`) REFERENCES `notifications` (`notification_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `notification_users_ibfk_user` FOREIGN KEY (`user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE CASCADE;

--
-- Constraints for table `user_document_preferences`
--
ALTER TABLE `user_document_preferences`
  ADD CONSTRAINT `user_document_preferences_ibfk_doc` FOREIGN KEY (`doc_id`) REFERENCES `dms_documents` (`doc_id`) ON DELETE CASCADE,
  ADD CONSTRAINT `user_document_preferences_ibfk_user` FOREIGN KEY (`user_id`) REFERENCES `dms_user` (`user_id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
